import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.actions import router as actions_router
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.config import Settings, load_settings
from app.db.workflow_migrations import migrate_workflow
from app.db.database import Database
from app.db.conversations import ConversationRepository
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.knowledge.bootstrap import (
    KnowledgeComponents,
    build_knowledge_components,
)
from app.model import ModelGateway, OpenAIModelGateway
from app.resource_lifecycle import close_resources
from app.schemas import AfterSalesResult, ExtractRequest
from app.services.actions import ActionService
from app.services.chat import ChatService
from app.services.workflow_chat import WorkflowChatService
from app.sessions import SessionGuard
from app.tools.executor import ToolExecutor
from app.workflow.bootstrap import build_workflow_dependencies
from app.workflow.checkpoints import CheckpointStore
from app.workflow.graph import WorkflowDependencies, build_workflow

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
    workflow_dependencies: WorkflowDependencies | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configuration = settings if settings is not None else load_settings()
        app.state.settings = configuration
        database = None
        model_gateway = gateway
        owned_resources: list[Any] = []
        cancellation: asyncio.CancelledError | None = None
        try:
            if model_gateway is None and chat_service is None:
                model_gateway = OpenAIModelGateway(configuration)
            if model_gateway is not None:
                owned_resources.append(model_gateway)
                app.state.gateway = model_gateway
            legacy_dependencies = knowledge_dependencies
            if legacy_dependencies is not None:
                owned_resources.extend(legacy_dependencies.resources)
            if chat_service is not None:
                app.state.chat_service = chat_service
            else:
                assert model_gateway is not None
                if configuration.database_url is None:
                    raise RuntimeError("DATABASE_URL is required for chat persistence")
                database = Database(configuration.database_url.get_secret_value())
                owned_resources.append(database)
                await database.check()
                if legacy_dependencies is not None and workflow_dependencies is None:
                    if (
                        legacy_dependencies.pipeline is None
                        or legacy_dependencies.low_confidence is None
                    ):
                        raise RuntimeError(
                            "knowledge dependencies require pipeline and low_confidence"
                        )
                    app.state.chat_service = ChatService(
                        configuration,
                        model_gateway,
                        ConversationRepository(database.sessions),
                        FaqRepository(database.sessions),
                        TicketRepository(database.sessions),
                        SessionGuard(configuration.max_sessions),
                        ToolExecutor(
                            configuration.tool_timeout_seconds,
                            configuration.tool_max_attempts,
                        ),
                        knowledge_pipeline=legacy_dependencies.pipeline,
                        low_confidence=legacy_dependencies.low_confidence,
                    )
                else:
                    migration = await migrate_workflow(database, check_only=True)
                    if migration.get("ready") != 1:
                        raise RuntimeError(
                            "workflow migration is not ready; run scripts/migrate_workflow.py"
                        )
                    components: KnowledgeComponents | None = None
                    dependencies = workflow_dependencies
                    if dependencies is None:
                        components = await build_knowledge_components(
                            configuration,
                            database,
                            model_gateway,
                            owned_resources,
                        )
                    checkpoint_store = CheckpointStore(configuration)
                    owned_resources.append(checkpoint_store)
                    saver = await checkpoint_store.open()
                    await checkpoint_store.check()
                    if dependencies is None:
                        assert components is not None
                        dependencies = build_workflow_dependencies(
                            configuration,
                            database,
                            model_gateway,
                            components,
                        )
                    graph = build_workflow(dependencies, saver)
                    workflow_service = WorkflowChatService(
                        configuration,
                        graph,
                        dependencies.conversations,
                        SessionGuard(configuration.max_sessions),
                    )
                    owned_resources.append(workflow_service)
                    app.state.chat_service = workflow_service
                    app.state.action_service = ActionService(
                        dependencies.actions,
                        dependencies.agent_dependencies.faq,
                        dependencies.agent_dependencies.tickets,
                        dependencies.agent_dependencies.executor,
                        configuration,
                    )
                    if components is not None:
                        app.state.knowledge_repository = components.repository
            if legacy_dependencies is not None:
                app.state.knowledge_repository = legacy_dependencies.repository
            yield
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:
            await close_resources(owned_resources, cancellation=cancellation)

    app = FastAPI(lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(actions_router)
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
