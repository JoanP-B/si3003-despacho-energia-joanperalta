# Corpus regulatorio (RAG)

Aquí se guardan los documentos que alimentan el RAG del agente LLM.

## Qué descargar

- **Resoluciones CREG** relevantes a almacenamiento de energía, mercado de energía mayorista y
  precio de bolsa. Fuente: sitio oficial de la CREG (creg.gov.co).
- **Documentos de XM** sobre operación del mercado y despacho. Fuente: xm.com.co /
  sinergox.xm.com.co.

Coloca los PDF (o HTML del gestor normativo) en esta carpeta. Los PDF están ignorados por
git (`.gitignore`) por tamaño; la tabla documenta su URL de origen para reproducibilidad.
Después de agregar o cambiar archivos, reconstruye el índice:

```bash
python -m src.rag.vectorstore
```

| Archivo | Título | URL de origen | ¿Indexado? |
|---------|--------|---------------|------------|
| `Creg098-2019.pdf` | Resolución CREG 098 de 2019: mecanismos para incorporar sistemas de almacenamiento (SAEB) | [PDF](https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/28669c6087eb69be0525846c006e9690/$FILE/Creg098-2019.pdf) | No: es un escaneo sin texto |
| `resolucion_creg_0098_2019.htm` | Resolución CREG 098 de 2019 (versión HTML con texto) | [HTML](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0098_2019.htm) | Sí |
| `Creg174-2021.pdf` | Resolución CREG 174 de 2021: autogeneración a pequeña escala y generación distribuida | [PDF](https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf) | Sí (67 páginas) |
