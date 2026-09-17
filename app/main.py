import asyncio
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.chat import router
from app.config import Settings, load_settings
from app.db.database import Database
from app.db.conversations import ConversationRepository
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.model import ModelGateway, OpenAIModelGateway
from app.schemas import AfterSalesResult, ExtractRequest
from app.services.chat import ChatService
from app.sessions import SessionGuard
from app.tools.executor import ToolExecutor


def _upstream_error() -> ServiceError:
    return ServiceError("UPSTREAM_ERROR", "模型服务暂时不可用", 502)


def _timeout_error() -> ServiceError:
    return ServiceError("UPSTREAM_TIMEOUT", "模型服务响应超时，请重试", 504)


def create_app(
    settings: Settings | None = None, gateway: ModelGateway | None = None,
    *, chat_service: ChatService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configuration = settings if settings is not None else load_settings()
        app.state.settings = configuration
        database = None
        model_gateway = gateway
        try:
            if model_gateway is None:
                model_gateway = OpenAIModelGateway(configuration)
            app.state.gateway = model_gateway
            if chat_service is None:
                if configuration.database_url is None:
                    raise RuntimeError("DATABASE_URL is required for chat persistence")
                database = Database(configuration.database_url.get_secret_value())
                await database.check()
                app.state.chat_service = ChatService(
                    configuration, model_gateway,
                    ConversationRepository(database.sessions),
                    FaqRepository(database.sessions), TicketRepository(database.sessions),
                    SessionGuard(configuration.max_sessions),
                    ToolExecutor(configuration.tool_timeout_seconds, configuration.tool_max_attempts),
                )
            else:
                app.state.chat_service = chat_service
            yield
        finally:
            try:
                if model_gateway is not None:
                    await model_gateway.aclose()
            finally:
                if database is not None:
                    await database.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

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
