# Corpus regulatorio (RAG)

Aquí se guardan los documentos que alimentan el RAG del agente LLM. Los dos documentos que se
indexan están versionados en el repositorio, así que el índice se construye sin descargas
manuales:

```bash
python -m src.rag.vectorstore
```

Después de agregar o cambiar archivos hay que reconstruir el índice con el mismo comando. Los
demás PDF de esta carpeta están ignorados por git (`.gitignore`) por tamaño; la tabla documenta
su URL de origen para reproducibilidad.

| Archivo | Título | URL de origen | ¿En el repositorio? | ¿Indexado? |
|---------|--------|---------------|---|------------|
| `resolucion_creg_0098_2019.htm` | Resolución CREG 098 de 2019 (versión HTML con texto) | [HTML](https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0098_2019.htm) | Sí | Sí |
| `Creg174-2021.pdf` | Resolución CREG 174 de 2021: autogeneración a pequeña escala y generación distribuida | [PDF](https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf) | Sí (3.4 MB) | Sí (67 páginas) |
| `Creg098-2019.pdf` | Resolución CREG 098 de 2019: mecanismos para incorporar sistemas de almacenamiento (SAEB) | [PDF](https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/28669c6087eb69be0525846c006e9690/$FILE/Creg098-2019.pdf) | No | No: es un escaneo sin texto |

**Vigencia:** el gestor normativo marca la CREG 098 de 2019 como *derogada por el artículo 108
de la Resolución 101 113 de 2026*. Se conserva como antecedente regulatorio de los SAEB.

Si alguna vez falta `Creg174-2021.pdf`, se puede descargar de nuevo:

```powershell
Invoke-WebRequest -OutFile corpus_regulacion\Creg174-2021.pdf -Uri 'https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf'
```

```bash
curl -L -o corpus_regulacion/Creg174-2021.pdf 'https://gestornormativo.creg.gov.co/Publicac.nsf/1c09d18d2d5ffb5b05256eee00709c02/c99b2f316a59fffb052587950077d9d1/$FILE/Creg174-2021.pdf'
```
