# ORBET Panel (oscuro + login)

## Requisitos
```
pip install fastapi uvicorn jinja2
```

## Ejecutar
```
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Abrir: http://localhost:8000

### Usuario/contraseña inicial
- Usuario: `admin`
- Contraseña: `admin123`

Puedes cambiarlos en **Config**.

### Conexión a tu ORBET
En **Config**, ajusta:
- Ruta DB ORBET (ingresos.db) — por defecto: ../ORBET_ACTUALIZADO/ingresos.db
- Carpeta capturas
- Carpeta tickets
