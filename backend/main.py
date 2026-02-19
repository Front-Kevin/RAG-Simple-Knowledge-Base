import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.document import router as document_router
from api.query import router as query_router
from api.settings import router as settings_router

app = FastAPI(title="RAG Knowledge Base", version="1.0.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": "".join(tb)})

# CORS 配置，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_router)
app.include_router(query_router)
app.include_router(settings_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------- 静态文件托管（Docker 打包模式） ----------
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    # 挂载 JS/CSS/图片等静态资源
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    # SPA catch-all：非 /api 路径都返回 index.html，由前端路由接管
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(_static_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_static_dir, "index.html"))
