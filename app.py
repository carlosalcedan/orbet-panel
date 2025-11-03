# app.py — ORBET PANEL (panel + login + reportes + API en tiempo real)
from __future__ import annotations
import os, json, io, csv, zipfile, secrets
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# ===================== RUTAS BASE =====================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DB_DIR = BASE_DIR / "db"

CAPTURAS_DIR = Path(os.getenv("ORBET_CAPTURAS_DIR", str(BASE_DIR / "capturas")))
TICKETS_DIR  = Path(os.getenv("ORBET_TICKETS_DIR",  str(BASE_DIR / "tickets")))

USERS_FILE = DB_DIR / "users.json"
SETTINGS_FILE = DB_DIR / "settings.json"

for d in (TEMPLATES_DIR, STATIC_DIR, DB_DIR, CAPTURAS_DIR, TICKETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ===================== APP =====================
app = FastAPI(title="ORBET Panel", version="1.0.0")
secret = os.getenv("ORBET_SECRET") or secrets.token_hex(32)
app.add_middleware(SessionMiddleware, secret_key=secret)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals.update(current_year=datetime.now().year)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ===================== UTILIDADES GENERALES =====================
def _safe_join(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        raise HTTPException(status_code=400, detail="Ruta inválida")
    return p

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _detect_tipo_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["persona","rostro","face","human"]): return "persona"
    if any(k in n for k in ["vehiculo","vehículo","carro","auto","camion","camión","truck","vehicle"]): return "vehiculo"
    if "animal" in n: return "animal"
    return "desconocido"

def _list_files(folder: Path, exts: tuple = (".jpg",".jpeg",".png",".gif",".webp",".bmp",".pdf",".txt")) -> List[Dict]:
    items: List[Dict] = []
    if not folder.exists():
        return items
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            items.append({
                "name": p.name,
                "rel": p.relative_to(folder).as_posix(),
                "path": str(p),
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime),
                "tipo": _detect_tipo_from_name(p.name)
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items

def _today_dir(base: Path) -> Path:
    d = base / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d

# ===================== USUARIOS =====================
def _load_users() -> List[Dict]:
    if not USERS_FILE.exists():
        default = [{"username": "carlos", "password": "orbet123", "role": "admin"}]
        USERS_FILE.write_text(json.dumps(default, indent=2, ensure_ascii=False), encoding="utf-8")
        return default
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_users(users: List[Dict]):
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")

def _verify_password(plain: str, user: Dict) -> bool:
    if "password_hash" in user:
        try:
            from passlib.hash import bcrypt
            return bcrypt.verify(plain, user["password_hash"])
        except Exception:
            return False
    return plain == user.get("password")

def _require_auth(request: Request):
    if not request.session.get("user"):
        raise HTTPException(status_code=401)

def auth_required(request: Request):
    _require_auth(request)
    return True

def role_required(request: Request, role: str = "admin"):
    _require_auth(request)
    user = request.session.get("user") or {}
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail="No autorizado")
    return True

# ===================== SETTINGS (con migración) =====================
def _settings_default():
    return {
        "telegram": {
            "enabled": False,
            "bot_token": "",
            "chat_id": "",
            "start": "21:00",
            "end": "05:00",
            "types": {"persona": True, "vehiculo": True, "animal": False},
            "send_image": True
        },
        "reports": {
            "enabled": True,
            "time": "23:55",
            "send_to_telegram": True
        }
    }

def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict):
            node = dst.get(k, {})
            if not isinstance(node, dict):
                node = {}
            dst[k] = _deep_merge(node, v)
        else:
            dst.setdefault(k, v)
    return dst

def _load_settings():
    default = _settings_default()
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(default, indent=2, ensure_ascii=False), encoding="utf-8")
        return default
    try:
        current = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = _deep_merge(current if isinstance(current, dict) else {}, default)
        SETTINGS_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        return merged
    except Exception:
        SETTINGS_FILE.write_text(json.dumps(default, indent=2, ensure_ascii=False), encoding="utf-8")
        return default

def _save_settings(s: dict):
    SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

# ===================== SALUD / HOME =====================
@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/ping")
def api_ping():
    return {"pong": True, "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/", response_class=HTMLResponse)
def home(_: Request):
    return """
    <html><head><title>ORBET PANEL</title></head>
    <body style="background:#0b0f14;color:#eaeef3;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh">
      <div style="text-align:center">
        <h1>✅ ORBET Panel activo</h1>
        <p><a href="/login" style="color:#7fb0ff">Ingresar</a> · <a href="/docs" style="color:#7fb0ff">API Docs</a> · <a href="/panel" style="color:#7fb0ff">Panel</a></p>
      </div>
    </body></html>
    """

# ===================== AUTH / NAVEGACIÓN =====================
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "message": None})

@app.post("/login")
def login_do(request: Request, username: str = Form(...), password: str = Form(...)):
    users = _load_users()
    u = next((x for x in users if x.get("username")==username), None)
    if not u or not _verify_password(password, u):
        return templates.TemplateResponse("login.html", {"request": request, "message": "Usuario o contraseña incorrectos."}, status_code=401)
    request.session["user"] = {"username": u["username"], "role": u.get("role","user")}
    return RedirectResponse(url="/panel", status_code=302)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)

# ===================== PANEL =====================
@app.get("/panel", response_class=HTMLResponse)
def panel(request: Request, _: bool = Depends(auth_required)):
    caps = _list_files(CAPTURAS_DIR, exts=(".jpg",".jpeg",".png",".webp",".bmp"))
    tics = _list_files(TICKETS_DIR,   exts=(".jpg",".jpeg",".png",".webp",".bmp",".pdf",".txt"))
    def c(items, tipo): return sum(1 for i in items if i["tipo"]==tipo)
    stats = {
        "capturas_total": len(caps),
        "tickets_total": len(tics),
        "personas": c(caps+tics,"persona"),
        "vehiculos": c(caps+tics,"vehiculo"),
        "animales": c(caps+tics,"animal")
    }
    return templates.TemplateResponse("panel.html", {"request": request, "stats": stats, "user": request.session.get("user")})

# ===================== CAPTURAS / TICKETS (filtros tipo + fecha) =====================
@app.get("/capturas", response_class=HTMLResponse)
def capturas_view(request: Request, tipo: str = "all", desde: str | None = None, hasta: str | None = None, _: bool = Depends(auth_required)):
    files = _list_files(CAPTURAS_DIR, exts=(".jpg",".jpeg",".png",".webp",".bmp"))
    if tipo in {"persona","vehiculo","animal"}:
        files = [f for f in files if f["tipo"] == tipo]
    d1 = _parse_date(desde); d2 = _parse_date(hasta) or (d1 if d1 else None)
    if d1 or d2:
        def in_range(xdate: date):
            if d1 and xdate < d1: return False
            if d2 and xdate > d2: return False
            return True
        files = [f for f in files if in_range(f["mtime"].date())]
    return templates.TemplateResponse("capturas.html", {"request": request, "files": files, "tipo": tipo, "desde": desde, "hasta": hasta})

@app.get("/tickets", response_class=HTMLResponse)
def tickets_view(request: Request, tipo: str = "all", desde: str | None = None, hasta: str | None = None, _: bool = Depends(auth_required)):
    files = _list_files(TICKETS_DIR, exts=(".jpg",".jpeg",".png",".webp",".bmp",".pdf",".txt"))
    if tipo in {"persona","vehiculo","animal"}:
        files = [f for f in files if f["tipo"] == tipo]
    d1 = _parse_date(desde); d2 = _parse_date(hasta) or (d1 if d1 else None)
    if d1 or d2:
        def in_range(xdate: date):
            if d1 and xdate < d1: return False
            if d2 and xdate > d2: return False
            return True
        files = [f for f in files if in_range(f["mtime"].date())]
    return templates.TemplateResponse("tickets.html", {"request": request, "files": files, "tipo": tipo, "desde": desde, "hasta": hasta})

# ===================== SERVIR ARCHIVOS (subcarpetas) =====================
@app.get("/capture/{relpath:path}")
def get_capture(relpath: str, _: bool = Depends(auth_required)):
    path = _safe_join(CAPTURAS_DIR, relpath)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Captura no encontrada")
    return FileResponse(str(path))

@app.get("/ticket/{relpath:path}")
def get_ticket(relpath: str, _: bool = Depends(auth_required)):
    path = _safe_join(TICKETS_DIR, relpath)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Ticket no encontrado")
    return FileResponse(str(path))

# ===================== API en tiempo real (garita → nube) =====================
def _check_token(token: str):
    expected = os.getenv("ORBET_API_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(status_code=401, detail="Token inválido")

def _save_upload(base_dir: Path, imagen: UploadFile, metadata: Optional[str]) -> Dict:
    day = _today_dir(base_dir)
    # nombre seguro
    stamp = datetime.now().strftime("%H%M%S")
    pure = Path(imagen.filename or f"file_{stamp}.bin").name
    out = day / f"{stamp}_{pure}"
    with open(out, "wb") as f:
        f.write(imagen.file.read())
    # guardar metadata opcional
    if metadata:
        try:
            md = json.loads(metadata)
        except Exception:
            md = {"raw": metadata}
        (out.with_suffix(out.suffix + ".json")).write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": out.relative_to(base_dir).as_posix()}

@app.post("/api/capturas")
def api_capturas(token: str = Query(...), imagen: UploadFile = File(...), metadata: str | None = Form(None)):
    _check_token(token)
    info = _save_upload(CAPTURAS_DIR, imagen, metadata)
    return {"ok": True, "type": "captura", **info}

@app.post("/api/tickets")
def api_tickets(token: str = Query(...), imagen: UploadFile = File(...), metadata: str | None = Form(None)):
    _check_token(token)
    info = _save_upload(TICKETS_DIR, imagen, metadata)
    return {"ok": True, "type": "ticket", **info}

# ===================== REPORTES =====================
def _recolectar_rows(desde: Optional[str], hasta: Optional[str], origen: str) -> List[Dict]:
    d1 = _parse_date(desde)
    d2 = _parse_date(hasta) or (d1 if d1 else None)

    rows: List[Dict] = []
    def take(items: List[Dict], fuente: str):
        for f in items:
            xdate = f["mtime"].date()
            if d1 and xdate < d1: 
                continue
            if d2 and xdate > d2: 
                continue
            rows.append({
                "fecha": xdate.strftime("%Y-%m-%d"),
                "hora": f["mtime"].strftime("%I:%M %p"),
                "tipo": f["tipo"],
                "fuente": fuente,
                "archivo": f["rel"]
            })

    if origen in ("ambos","capturas"):
        take(_list_files(CAPTURAS_DIR, exts=(".jpg",".jpeg",".png",".webp",".bmp")), "captura")
    if origen in ("ambos","tickets"):
        take(_list_files(TICKETS_DIR, exts=(".jpg",".jpeg",".png",".webp",".bmp",".pdf",".txt")), "ticket")

    rows.sort(key=lambda r: (r["fecha"], r["hora"], r["tipo"]))
    return rows

def _generar_reporte(rows: List[Dict]) -> Path:
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        from openpyxl import Workbook  # type: ignore
        wb = Workbook(); ws = wb.active; ws.title = "detecciones"
        ws.append(["Fecha", "Hora", "Tipo", "Fuente", "Archivo"])
        for r in rows: ws.append([r["fecha"], r["hora"], r["tipo"], r["fuente"], r["archivo"]])
        path = reports_dir / f"reporte_{stamp}.xlsx"
        wb.save(str(path)); return path
    except Exception:
        path = reports_dir / f"reporte_{stamp}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["Fecha","Hora","Tipo","Fuente","Archivo"])
            for r in rows: w.writerow([r["fecha"], r["hora"], r["tipo"], r["fuente"], r["archivo"]])
        return path

@app.get("/reporte", response_class=HTMLResponse)
def reporte_page(request: Request, _: bool = Depends(auth_required)):
    return templates.TemplateResponse("reporte.html", {"request": request, "message": None})

@app.post("/reporte")
def generar_reporte(request: Request, desde: str = Form(None), hasta: str = Form(None), origen: str = Form("ambos")):
    rows = _recolectar_rows(desde, hasta, origen)
    used_path = _generar_reporte(rows)
    return FileResponse(str(used_path), filename=used_path.name)

@app.get("/reporte/hoy")
def reporte_hoy(_: Request, __: bool = Depends(auth_required)):
    today = datetime.now().strftime("%Y-%m-%d")
    rows = _recolectar_rows(today, today, "ambos")
    used_path = _generar_reporte(rows)
    return FileResponse(str(used_path), filename=used_path.name)

# ===================== DESCARGAS (ZIP por día) =====================
@app.get("/descargas", response_class=HTMLResponse)
def descargas_page(request: Request, _: bool = Depends(auth_required)):
    return templates.TemplateResponse("descargas.html", {"request": request})

@app.post("/descargas")
def descargar_zip(request: Request, carpeta: str = Form("capturas"), dia: str = Form(...)):
    base = CAPTURAS_DIR if carpeta == "capturas" else TICKETS_DIR
    day_dir = base / dia
    if not day_dir.exists() or not day_dir.is_dir():
        return templates.TemplateResponse("descargas.html", {"request": request, "message": "No existe esa carpeta de día."}, status_code=400)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in day_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(base))
    buf.seek(0)
    zipname = f"{carpeta}_{dia}.zip"
    return FileResponse(buf, media_type="application/zip", filename=zipname)

# ===================== AJUSTES =====================
@app.get("/ajustes", response_class=HTMLResponse)
def ajustes_page(request: Request, _: bool = Depends(role_required)):
    settings = _load_settings()
    return templates.TemplateResponse("ajustes.html", {"request": request, "s": settings, "message": None})

@app.post("/ajustes", response_class=HTMLResponse)
def ajustes_save(
    request: Request,
    tg_enabled: str = Form("off"),
    tg_bot: str = Form(""),
    tg_chat: str = Form(""),
    tg_start: str = Form("21:00"),
    tg_end: str = Form("05:00"),
    tg_persona: str = Form("on"),
    tg_vehiculo: str = Form("on"),
    tg_animal: str = Form("off"),
    tg_send_img: str = Form("on"),
    rep_enabled: str = Form("on"),
    rep_time: str = Form("23:55"),
    rep_to_tg: str = Form("on"),
    _: bool = Depends(role_required)
):
    s = _load_settings()
    s["telegram"]["enabled"] = (tg_enabled == "on")
    s["telegram"]["bot_token"] = tg_bot.strip()
    s["telegram"]["chat_id"] = tg_chat.strip()
    s["telegram"]["start"] = tg_start.strip() or "21:00"
    s["telegram"]["end"] = tg_end.strip() or "05:00"
    s["telegram"]["types"] = {
        "persona": tg_persona == "on",
        "vehiculo": tg_vehiculo == "on",
        "animal": tg_animal == "on",
    }
    s["telegram"]["send_image"] = (tg_send_img == "on")
    s["reports"]["enabled"] = (rep_enabled == "on")
    s["reports"]["time"] = rep_time.strip() or "23:55"
    s["reports"]["send_to_telegram"] = (rep_to_tg == "on")

    _save_settings(s)
    return templates.TemplateResponse("ajustes.html", {"request": request, "s": s, "message": "Ajustes guardados."})

# ===================== USUARIOS (admin) =====================
@app.get("/usuarios", response_class=HTMLResponse)
def users_page(request: Request, _: bool = Depends(role_required)):
    return templates.TemplateResponse("usuarios.html", {"request": request, "users": _load_users(), "message": None})

@app.post("/usuarios/crear")
def users_create(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form("user"), _: bool = Depends(role_required)):
    users = _load_users()
    if any(u["username"] == username for u in users):
        return templates.TemplateResponse("usuarios.html", {"request": request, "users": users, "message": "Usuario ya existe."}, status_code=400)
    users.append({"username": username, "password": password, "role": role})
    _save_users(users)
    return RedirectResponse(url="/usuarios", status_code=302)

@app.post("/usuarios/borrar")
def users_delete(request: Request, username: str = Form(...), _: bool = Depends(role_required)):
    users = [u for u in _load_users() if u["username"] != username]
    _save_users(users)
    return RedirectResponse(url="/usuarios", status_code=302)

# ===================== TEMPLATES MÍNIMOS =====================
def _ensure_min_template(filename: str, content: str):
    p = TEMPLATES_DIR / filename
    if not p.exists():
        p.write_text(content, encoding="utf-8")

_ensure_min_template("base.html", """<!doctype html>
<html lang="es"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% block title %}ORBET{% endblock %}</title>
<style>
body{background:#0b0f14;color:#eaeef3;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial;margin:0}
header,footer{padding:14px 18px;background:#0f1621;border-bottom:1px solid #1d2836}
.container{max-width:1100px;margin:0 auto;padding:18px}
a{color:#7fb0ff;text-decoration:none}.nav a{margin-right:12px}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #1d2836}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;background:#172233;border:1px solid #264365;margin-left:6px;font-size:12px}
.pill{padding:2px 8px;border-radius:10px;background:#1a2534;font-size:12px}
button{padding:8px 12px;border-radius:8px;border:1px solid #27425f;background:#1b2737;color:#eaeef3;cursor:pointer}
input,select{background:#0e141b;color:#eaeef3;border:1px solid #223044;border-radius:8px;padding:8px}
</style></head><body>
<header><div class="container nav"><strong>ORBET</strong>
<span style="margin-left:16px">
  <a href="/panel">Panel</a>
  <a href="/capturas">Capturas</a>
  <a href="/tickets">Tickets</a>
  <a href="/reporte">Reporte</a>
  <a href="/descargas">Descargas</a>
  <a href="/ajustes">Ajustes</a>
  <a href="/usuarios">Usuarios</a>
</span>
<span style="float:right"><a href="/logout">Salir</a></span>
</div></header>
<div class="container">{% block content %}{% endblock %}</div>
<footer style="border-top:1px solid #1d2836"><div class="container">© {{ current_year }} ORBET</div></footer>
</body></html>
""")

_ensure_min_template("login.html", """{% extends "base.html" %}{% block title %}Login - ORBET{% endblock %}
{% block content %}
<h2>Acceso</h2>
<form method="post" action="/login" style="max-width:360px">
<label>Usuario</label><br><input name="username" required style="width:100%"><br><br>
<label>Contraseña</label><br><input type="password" name="password" required style="width:100%"><br><br>
<button type="submit">Ingresar</button>
</form>
{% if message %}<p style="color:#f39c12">{{ message }}</p>{% endif %}
{% endblock %}
""")

_ensure_min_template("panel.html", """{% extends "base.html" %}{% block title %}Panel - ORBET{% endblock %}
{% block content %}
<h2>Panel</h2>
<p>Bienvenido {{ user.username if user else '' }}.</p>
<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;max-width:800px;margin-bottom:16px">
<div class="pill">Capturas: <strong>{{ stats.capturas_total }}</strong></div>
<div class="pill">Tickets: <strong>{{ stats.tickets_total }}</strong></div>
<div class="pill">Personas: <strong>{{ stats.personas }}</strong></div>
<div class="pill">Vehículos: <strong>{{ stats.vehiculos }}</strong></div>
<div class="pill">Animales: <strong>{{ stats.animales }}</strong></div>
</div>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 20px">
  <a href="/reporte/hoy"><button>📄 Descargar reporte de HOY</button></a>
  <a href="/reporte"><button>⚙️ Reporte avanzado</button></a>
  <a href="/descargas"><button>🗜️ Crear ZIP por fecha</button></a>
</div>
{% endblock %}
""")

_ensure_min_template("capturas.html", """{% extends "base.html" %}{% block title %}Capturas - ORBET{% endblock %}
{% block content %}
<h2>Capturas</h2>
<p>Filtro:
<a href="/capturas?tipo=all{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Todos</a>
<a href="/capturas?tipo=persona{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Persona</a>
<a href="/capturas?tipo=vehiculo{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Vehículo</a>
<a href="/capturas?tipo=animal{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Animal</a></p>

<form method="get" action="/capturas" style="margin:10px 0">
  <input type="hidden" name="tipo" value="{{ tipo }}">
  Desde: <input type="date" name="desde" value="{{ desde or '' }}">
  Hasta: <input type="date" name="hasta" value="{{ hasta or '' }}">
  <button type="submit">Filtrar</button>
  <a href="/capturas?tipo={{ tipo }}"><button type="button">Limpiar</button></a>
</form>

<table><thead><tr><th>Archivo</th><th>Tipo</th><th>Tamaño</th><th>Fecha</th><th>Ver</th></tr></thead><tbody>
{% for f in files %}
<tr>
<td>{{ f.rel }}</td>
<td>{{ f.tipo }}</td>
<td>{{ (f.size/1024)|round(1) }} KB</td>
<td>{{ f.mtime.strftime("%Y-%m-%d %H:%M:%S") }}</td>
<td><a href="/capture/{{ f.rel }}" target="_blank">Abrir</a></td>
</tr>
{% else %}<tr><td colspan="5">No hay archivos.</td></tr>{% endfor %}
</tbody></table>
{% endblock %}
""")

_ensure_min_template("tickets.html", """{% extends "base.html" %}{% block title %}Tickets - ORBET{% endblock %}
{% block content %}
<h2>Tickets</h2>
<p>Filtro:
<a href="/tickets?tipo=all{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Todos</a>
<a href="/tickets?tipo=persona{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Persona</a>
<a href="/tickets?tipo=vehiculo{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Vehículo</a>
<a href="/tickets?tipo=animal{% if desde %}&desde={{desde}}{% endif %}{% if hasta %}&hasta={{hasta}}{% endif %}" class="tag">Animal</a></p>

<form method="get" action="/tickets" style="margin:10px 0">
  <input type="hidden" name="tipo" value="{{ tipo }}">
  Desde: <input type="date" name="desde" value="{{ desde or '' }}">
  Hasta: <input type="date" name="hasta" value="{{ hasta or '' }}">
  <button type="submit">Filtrar</button>
  <a href="/tickets?tipo={{ tipo }}"><button type="button">Limpiar</button></a>
</form>

<table><thead><tr><th>Archivo</th><th>Tipo</th><th>Tamaño</th><th>Fecha</th><th>Ver</th></tr></thead><tbody>
{% for f in files %}
<tr>
<td>{{ f.rel }}</td>
<td>{{ f.tipo }}</td>
<td>{{ (f.size/1024)|round(1) }} KB</td>
<td>{{ f.mtime.strftime("%Y-%m-%d %H:%M:%S") }}</td>
<td><a href="/ticket/{{ f.rel }}" target="_blank">Abrir</a></td>
</tr>
{% else %}<tr><td colspan="5">No hay archivos.</td></tr>{% endfor %}
</tbody></table>
{% endblock %}
""")

_ensure_min_template("reporte.html", """{% extends "base.html" %}{% block title %}Reporte - ORBET{% endblock %}
{% block content %}
<h2>Reporte</h2>
<form method="post" action="/reporte" style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
  <label>Desde: <input type="date" name="desde"></label>
  <label>Hasta: <input type="date" name="hasta"></label>
  <label>Origen:
    <select name="origen">
      <option value="ambos">Ambos</option>
      <option value="capturas">Solo Capturas</option>
      <option value="tickets">Solo Tickets</option>
    </select>
  </label>
  <button type="submit">Generar</button>
  <a href="/reporte/hoy"><button type="button">Reporte de HOY</button></a>
</form>
{% endblock %}
""")

_ensure_min_template("usuarios.html", """{% extends "base.html" %}{% block title %}Usuarios - ORBET{% endblock %}
{% block content %}
<h2>Usuarios (admin)</h2>
{% if message %}<p style="color:#f39c12">{{ message }}</p>{% endif %}
<form method="post" action="/usuarios/crear" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px">
  <input name="username" placeholder="usuario" required>
  <input type="password" name="password" placeholder="contraseña" required>
  <select name="role"><option value="user">user</option><option value="admin">admin</option></select>
  <button type="submit">Crear</button>
</form>
<table><thead><tr><th>Usuario</th><th>Rol</th><th></th></tr></thead><tbody>
{% for u in users %}
<tr>
<td>{{ u.username }}</td>
<td>{{ u.role }}</td>
<td>
  {% if u.username != 'carlos' %}
  <form method="post" action="/usuarios/borrar" onsubmit="return confirm('¿Borrar {{u.username}}?')">
    <input type="hidden" name="username" value="{{u.username}}">
    <button>Eliminar</button>
  </form>
  {% endif %}
</td>
</tr>
{% else %}<tr><td colspan="3">Sin usuarios.</td></tr>{% endfor %}
</tbody></table>
{% endblock %}
""")

_ensure_min_template("descargas.html", """{% extends "base.html" %}{% block title %}Descargas - ORBET{% endblock %}
{% block content %}
<h2>Descargas (ZIP por día)</h2>
{% if message %}<p style="color:#f39c12">{{ message }}</p>{% endif %}
<form method="post" action="/descargas" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
  <label>Carpeta:
    <select name="carpeta">
      <option value="capturas">capturas</option>
      <option value="tickets">tickets</option>
    </select>
  </label>
  <label>Día (YYYY-MM-DD):
    <input name="dia" placeholder="2025-10-29" required>
  </label>
  <button type="submit">Crear ZIP</button>
</form>
<p style="opacity:.7;margin-top:8px">Nota: Debe existir la subcarpeta con ese día dentro de la carpeta elegida.</p>
{% endblock %}
""")
# ===================== FIN app.py =====================

