import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from sqlalchemy import inspect

from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.config import Settings, load_settings
from app.db.database import Database
from app.db.conversations import ConversationRepository
from app.db.faq import FaqRepository
from app.db.knowledge import KnowledgeRepository
from app.db.low_confidence import LowConfidenceRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.model import ModelGateway, OpenAIModelGateway
from app.knowledge.calibration import load_runtime_calibration
from app.knowledge.corpus import corpus_fingerprint
from app.knowledge.local_models import LocalModels
from app.knowledge.milvus_store import MilvusStore
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.query import QueryNormalizer
from app.knowledge.retrieval import KnowledgeRetriever
from app.schemas import AfterSalesResult, ExtractRequest
from app.services.chat import ChatService
from app.sessions import SessionGuard
from app.tools.executor import ToolExecutor


_KNOWLEDGE_TABLES = frozenset(
    {"knowledge_chunks", "qa_extraction_staging", "low_confidence_questions"}
)


@dataclass(frozen=True)
class KnowledgeDependencies:
    repository: Any
    pipeline: Any | None = None
    low_confidence: Any | None = None
    resources: tuple[Any, ...] = ()


def _upstream_error() -> ServiceError:
    return ServiceError("UPSTREAM_ERROR", "模型服务暂时不可用", 502)


def _timeout_error() -> ServiceError:
    return ServiceError("UPSTREAM_TIMEOUT", "模型服务响应超时，请重试", 504)


def create_app(
    settings: Settings | None = None, gateway: ModelGateway | None = None,
    *, chat_service: ChatService | None = None,
    knowledge_dependencies: KnowledgeDependencies | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configuration = settings if settings is not None else load_settings()
        app.state.settings = configuration
        database = None
        model_gateway = gateway
        owned_resources: list[Any] = []
        try:
            if model_gateway is None:
                model_gateway = OpenAIModelGateway(configuration)
            owned_resources.append(model_gateway)
            app.state.gateway = model_gateway
            dependencies = knowledge_dependencies
            if dependencies is not None:
                owned_resources.extend(dependencies.resources)
            if chat_service is None:
                if configuration.database_url is None:
                    raise RuntimeError("DATABASE_URL is required for chat persistence")
                database = Database(configuration.database_url.get_secret_value())
                owned_resources.append(database)
                await database.check()
                if dependencies is None and hasattr(
                    model_gateway, "create_knowledge_gateway"
                ):
                    dependencies = await _production_knowledge_dependencies(
                        configuration,
                        database,
                        model_gateway,
                        owned_resources,
                    )
                if dependencies is not None and (
                    dependencies.pipeline is None
                    or dependencies.low_confidence is None
                ):
                    raise RuntimeError(
                        "knowledge dependencies require pipeline and low_confidence"
                    )
                knowledge_kwargs = (
                    {}
                    if dependencies is None
                    else {
                        "knowledge_pipeline": dependencies.pipeline,
                        "low_confidence": dependencies.low_confidence,
                    }
                )
                app.state.chat_service = ChatService(
                    configuration, model_gateway,
                    ConversationRepository(database.sessions),
                    FaqRepository(database.sessions), TicketRepository(database.sessions),
                    SessionGuard(configuration.max_sessions),
                    ToolExecutor(configuration.tool_timeout_seconds, configuration.tool_max_attempts),
                    **knowledge_kwargs,
                )
            else:
                app.state.chat_service = chat_service
            if dependencies is not None:
                app.state.knowledge_repository = dependencies.repository
            yield
        finally:
            await _close_resources(owned_resources)

    app = FastAPI(lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(knowledge_router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return resources.files("app").joinpath("web/index.html").read_text(encoding="utf-8")

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(
            {"error": {
                "code": "SESSION_NOT_FOUND" if error.code == "CONVERSATION_NOT_FOUND" else error.code,
                "message": error.message,
            }},
            status_code=error.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Validation details can echo private customer input; expose only the contract.
        return JSONResponse(
            {"error": {"code": "INVALID_REQUEST", "message": "请求参数无效"}},
            status_code=422,
        )

    @app.post("/api/after-sales/extract", response_model=AfterSalesResult)
    async def extract(body: ExtractRequest, request: Request) -> AfterSalesResult:
        try:
            async with asyncio.timeout(request.app.state.settings.request_timeout_seconds):
                return await request.app.state.gateway.extract(body.description)
        except TimeoutError as exc:
            raise _timeout_error() from exc
        except ServiceError:
            raise
        except Exception as exc:
            raise _upstream_error() from exc

    return app


async def _production_knowledge_dependencies(
    settings: Settings,
    database: Database,
    model_gateway: ModelGateway,
    owned_resources: list[Any],
) -> KnowledgeDependencies:
    store = MilvusStore(settings)
    local_models = LocalModels(settings)
    owned_resources.extend((store, local_models))
    await _check_knowledge_tables(database)
    await store.prepare_existing_collection()

    repository = KnowledgeRepository(database.sessions)
    chunks = await repository.list_all()
    if not chunks:
        raise RuntimeError("knowledge corpus is empty; run scripts/init_knowledge.py")
    artifact = load_runtime_calibration(
        settings,
        corpus_fingerprint=corpus_fingerprint(chunks),
    )
    await _warmup_local_models(local_models)

    factory = getattr(model_gateway, "create_knowledge_gateway", None)
    if factory is None:
        raise RuntimeError("model gateway cannot create the knowledge gateway")
    knowledge_gateway = factory()
    normalizer = QueryNormalizer(knowledge_gateway)
    retriever = KnowledgeRetriever(repository, store, local_models)
    pipeline = KnowledgePipeline(
        normalizer,
        retriever,
        knowledge_gateway,
        artifact.threshold,
    )
    return KnowledgeDependencies(
        repository=repository,
        pipeline=pipeline,
        low_confidence=LowConfidenceRepository(database.sessions),
    )


async def _check_knowledge_tables(database: Database) -> None:
    async with database.engine.connect() as connection:
        names = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
    missing = sorted(_KNOWLEDGE_TABLES - names)
    if missing:
        raise RuntimeError(
            "knowledge tables are missing: "
            + ", ".join(missing)
            + "; run schema setup"
        )


async def _warmup_local_models(local_models: LocalModels) -> None:
    task = asyncio.create_task(asyncio.to_thread(local_models.warmup))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain_task(task)
        raise


async def _drain_task(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            return
    if not task.cancelled():
        try:
            task.exception()
        except BaseException:
            pass


async def _close_resources(resources_to_close: list[Any]) -> None:
    first_error: Exception | None = None
    for resource in reversed(resources_to_close):
        close = getattr(resource, "aclose", None)
        if close is None:
            continue
        try:
            await close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
