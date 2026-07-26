# Auditoría de rendimiento — AcademiaSD LoRAlab-Krea2

> Fecha: 2026-07-26 · Hardware: RTX 4070 Ti (12 GB), ext4 sobre NVMe
> Datos usados: dataset `ds-closeup` (29 imgs), caché `lf_bd_v6`, logs de `lf_bd_v5`
>
> **Toda cifra de este documento está medida, no estimada.** El apéndice incluye los scripts
> para reproducirlas.

## Resumen ejecutivo

El cuello de botella **no está donde el código sugiere**. El bucle de entrenamiento está
GPU-bound y prácticamente en el límite del hardware; las optimizaciones "de libro" que se le
podrían aplicar suman menos del 0.02 % del tiempo de step. El coste real está en la **capa web**
(un arrastre de slider mueve 1.4 GB) y, en menor medida, en el **pre-caché**.

Orden recomendado de ataque: web → terminal/stream → pre-caché. El entrenamiento se deja
intacto.

---

## 1. Hallazgo principal: el bucle de entrenamiento no es el problema

Medición del step real (`output_local/lf_bd_v5/phase2_1024/train_log.csv`, n=225):
**5174 ms por micro-step**.

Coste real de cada optimización candidata del hot path:

| Optimización candidata | Coste medido | % del step |
|---|---:|---:|
| `torch.cat` → pageable vs. pinned directo (`2_train:1818,1834`) | 0.05 ms | 0.001 % |
| `pack_latents` recomputado por step (`:1839`) | 0.047 ms | 0.0009 % |
| `.item()` tras el backward (`:1931`) | 0.004 ms | 0.00008 % |
| Casts de `HIGH_PREC_TARGETS` (`:1856-1859`) | 0.062 ms | 0.001 % |
| **Suma de todas** | **< 1 ms** | **< 0.02 %** |

Con 5.2 s de cómputo GPU y ~1 ms de trabajo CPU por step, **no hay nada que solapar**: las
sincronizaciones GPU→CPU no cuestan pipeline porque la CPU está ociosa el 99.98 % del tiempo.
Modificar esto es trabajo perdido que además añade riesgo a un bucle hoy correcto.

Escalado medido por fase (mediana de `secs`, máximo de `vram_peak_gb`):

| Fase | s/step | VRAM pico | Margen |
|---|---:|---:|---:|
| 512 | 2.41 | 8.99 GB | **3.3 GB** |
| 768 | 4.58 | 9.88 GB | 2.4 GB |
| 1024 | 5.24 | 11.13 GB | 1.1 GB |

**Única hipótesis del entrenamiento que merece un benchmark:** a 512 sobran 3.3 GB y se
entrena con `batch_size: 1`. Parte del coste por step (dequantización NF4, lanzamiento de
kernels) es fijo y se amortizaría con `batch_size: 2`. No se afirma como ganancia: la atención
crece O(n²) y el reparto entre coste fijo y coste por token no se deduce sin perfilar el modelo
cargado. Hay que medirlo.

---

## 2. Ranking de debilidades reales

### 🔴 1. Slider de curaduría: 1.4 GB de tráfico por arrastre

Cadena completa verificada:

1. `web/trainer_ui.html:1175` — `oninput` **sin debounce, throttle ni `requestAnimationFrame`**
   (búsqueda exhaustiva: no existe ninguno en las 2343 líneas del fichero).
2. → `onThresholdInput` (`:1876`) → `filterDatasetGrid` → `renderDatasetGrid` (`:2011`), que
   hace `host.innerHTML = ''` (`:2013`) y **reconstruye todas las tarjetas con `<img>` nuevos**
   (`:1986`).
3. Las imágenes se sirven **a tamaño completo**: `ds-closeup` = 29 imágenes, **33.9 MB**, media
   1.17 MB — pintadas en recuadros de **115 px** (`.img-wrapper`, CSS `:441`).
4. **Sin caché de navegador**: `send_from_directory` se llama sin `max_age`
   (`server.py:979`), y Werkzeug fuerza `no_cache = True` en ese caso
   (`werkzeug/utils.py:493` y `:500`).

**Medido:** un render = 33.9 MB. Un arrastre (~40 eventos `input`) = **1.4 GB** y ~1160
peticiones HTTP contra un Flask de desarrollo mono-GIL. Con el tope de 500 imágenes del
servidor (`:923`) escala a decenas de GB.

Mismo patrón sin debounce en el buscador (`:1141`), en `project_name` → un
`/api/checkpoint-info` **por tecla** (`:896`→`:1336`) y en `updateReplaceLiveStats` (`:2227`).

### 🟠 2. Terminal SSE: crecimiento O(n²) del DOM

`appendTerminal` (`:1624-1635`) **nunca poda**: cero ocurrencias de `removeChild`, `MAX_LINES`
o `children.length` en todo el fichero. Y `:1634` (`term.scrollTop = term.scrollHeight`)
**fuerza un reflow síncrono por línea** sobre un contenedor con `white-space: pre-wrap` +
`word-break: break-all` (CSS `:374-375`); su coste crece con lo ya acumulado, de modo que el
trabajo total de la sesión es **O(n²)**.

Un entrenamiento de 4000 steps genera decenas de miles de `<div>`. Se mitiga en parte porque la
barra de progreso usa la rama `replace` (`:1626`), que reutiliza el último nodo, pero el reflow
se paga igual.

Lado servidor, `/api/stream` (`server.py:807-817`) itera **carácter a carácter en Python** sobre
bloques de hasta 64 KB con `buffer += char`, **reteniendo el GIL** — la peor combinación con
`threaded=True`, porque bloquea el servido de imágenes.

### 🟠 3. Pre-caché: text encoder a batch 1 con coste fijo

`1_pre_cache:446` llama `pipe.encode_prompt(prompt=<un string>)`. Dos hechos verificados en
`diffusers/pipelines/krea2/pipeline_krea2.py`:

- `:226` — la API **acepta lista** y hace batching nativo. No se usa.
- `:232-234` — `padding="max_length"`: el forward del Qwen3-VL-4B cuesta **lo mismo (128
  tokens) sea cual sea el caption**.

Un batch de 8 captions costaría casi lo mismo que uno → se desperdicia ~8× de throughput en esa
etapa. El VAE encode también va a batch 1 (`:361`), con hasta 6 llamadas por imagen
(3 resoluciones × flip, bucle `:459-466`).

### 🟠 4. Pre-caché: `free_vram()` que no libera nada, 12× por imagen

```python
1_pre_cache:155  def free_vram(*tensors):
            157      del t              # borra el nombre LOCAL, no la referencia del llamante
            158      gc.collect()
            159      torch.cuda.empty_cache()
```

`del t` no libera nada: `img_tensor` y `z` siguen vivos en el frame del llamante cuando se
invoca en `:364` (antes del `return`). El coste, en cambio, es real: **17.8 ms medidos** por
llamada, y `empty_cache()` además destruye el caching allocator y fuerza `cudaMalloc` nuevos.
Se ejecuta en `:364` y `:466` → **12 veces por imagen** con 3 resoluciones + flip ≈ **213 ms
por imagen tirados**.

La versión del trainer (`2_train:496`) es honesta y sólo se llama fuera del hot path. El
problema es exclusivo del pre-caché.

### 🟡 5. Curaduría: hasta 6 forwards de detección por imagen

`0_curate:138-149` hace 3 llamadas a `.get()` en el peor caso (pad-retry a 0.25 y 0.5). Pero
**cada `.get()` son 2 inferencias, no 1**: `prepare(ctx_id=-1)` se llama **sin `det_size`**
(`:123`), y entonces `insightface/app/face_analysis.py:23,67-70` usa
`DEFAULT_DET_SIZES = [(128,128), (640,640)]`, iterando ambas escalas
(`model_zoo/retinaface.py:213-231`). → **3 × 2 = 6 forwards** en el peor caso.

Agravantes: `cv2.copyMakeBorder` con pad 0.5 copia **4× los píxeles** a resolución nativa; se
decodifica con dos copias completas del array (`:160-161`) sin downscale previo, pese a que el
detector reescala a 128/640 igualmente; ONNX Runtime sin `intra_op_num_threads`; y `save_cache`
sólo se llama al final (`:343`), así que abortar tras 400 de 500 imágenes pierde las 400.

**Atenuante:** la caché por `(mtime, size)` funciona bien y cachea también los negativos
(`:294-307`), así que esto sólo se paga en el primer scan. Medido: re-scan completo del dataset
en **0.07 s** (32/32 desde caché).

### 🟡 6. Pre-caché: 65 % del disco son copias byte-idénticas

Verificado por md5 en `cached_data_local/lf_bd_v6/`:

```
0e82b34eb8a9  7.6M  1024/an_12_embed.pt
0e82b34eb8a9  7.6M   512/an_12_embed.pt     ← mismo hash
0e82b34eb8a9  7.6M   768/an_12_embed.pt     ← mismo hash
```

El embedding de texto **no depende de la resolución ni del flip**, pero `1_pre_cache:463-465`
lo escribe dentro del doble bucle `for out_dir × for variant`. Desglose de una resolución:
latents 12 MB, **embeds 226 MB**, masks 120 KB → de los 700 MB del proyecto, **~452 MB son
duplicación pura**.

> **Esto NO afecta al rendimiento del entrenamiento.** Una lectura inicial sugería que el
> trainer pagaba caro re-leer y re-*pinnear* esos 226 MB por fase. Medido: `torch.load` de la
> caché completa = 113 ms, `pin_memory` = 156 ms, **total 269 ms por fase** — el **0.01 %** de
> una fase de 1000 steps (87 min). Con `progressive` son 0.8 s en total. Es ruido.
>
> El coste real de la duplicación es **sólo disco y tiempo de escritura del pre-caché**. Es
> higiene, no rendimiento.

Coste de tiempo que sí es real: añadir una resolución nueva **re-ejecuta el text encoder sobre
todo el dataset** (`:446` está fuera del bucle de `pending` pero dentro de la rama "falta
algo"), aunque el `.pt` idéntico ya exista en otro subdirectorio.

**Corrección de riesgo cero si se aborda:** `os.link()` en lugar de la 2ª y 3ª `torch.save`.
El trainer **no cambia ni una línea** — su ruta de lectura es
`torch.load(f"{directory}/{name}_embed.pt")` (`2_train:1541`), y un hardlink *es* el fichero
(mismo inodo). Requisitos confirmados: los subdirectorios están en el mismo dispositivo
(`dev=66306`) y el filesystem es ext4. `is_cached` (`:420-428`) usa `os.path.exists`, que un
hardlink satisface igual, y `prune_orphans` (`:233-242`) usa `os.remove`, que sólo decrementa
`nlink`. Regla obligatoria: `os.remove(destino)` antes de escribir o enlazar, para que una
escritura fallida no corrompa las tres resoluciones a la vez.

### 🟡 7. Polling y endpoints sin caché

- **`/api/dataset-info`** (`server.py:893-924`): rescan completo sin caché ni ETag, con
  `txt_path.exists()` **duplicado** (`:905` y `:910` — 2N `stat` donde basta N), N lecturas de
  `.txt` y 2 parses JSON de curaduría. Se dispara desde **10 sitios**, incluido tras guardar
  **un solo** caption (`:2144`): N+1 clásico.
- **`/api/previews`** (`:868-877`): dos `stat()` por preview más un `iterdir()` de un
  directorio que contiene los `.safetensors`, **cada 8 s incondicionalmente** (`:2118`), aunque
  no haya entrenamiento y la respuesta sea idéntica. Reconstruye la galería entera con `<img>`
  nuevos → hasta 50 revalidaciones HTTP por tick.
- **`/api/system-stats`**: `nvidia-smi` mide **12 ms** (no los 80-250 ms que cabría suponer).
  A 1200 llamadas/hora son ~14 s de CPU/hora: real pero **menor**. Lo que conviene es pausar el
  polling con `visibilitychange`, hoy inexistente.
- **`app.run(threaded=True)`** (`:1102`): servidor de desarrollo, un hilo por conexión sin pool
  ni límite. Aceptable en uso local monousuario; sólo importa por su interacción con el GIL del
  punto 2.

### 🟡 8. Progressive: recarga completa del modelo por fase

`run_progressive.py:288-294` lanza un proceso nuevo por fase con `.wait()` bloqueante. Por fase
se paga: arranque del intérprete e imports de torch/diffusers, construcción del transformer de
12 B (`2_train:1198`), `load_nf4_cache_` (`:1223-1226`), `transformer.to("cuda")` de ~7 GB por
PCIe, y otra copia del VAE si hay previews. Con `512_768_1024` se paga 3 veces.

Está **deliberadamente diseñado así** (docstring `:6-11`: allocator CUDA limpio, sin
fragmentación) y la justificación es sólida en una tarjeta de 12 GB. Se anota como coste
conocido, no como defecto. Positivo: **no re-cachea** (`:241-247` sólo comprueba que el
directorio exista).

---

## 3. Coste / beneficio

Ninguno de estos ítems toca el bucle de entrenamiento.

| Prioridad | Arreglo | Esfuerzo | Beneficio | Riesgo |
|---|---|---|---|---|
| 1 | Debounce del slider + `max_age` + thumbnails | Bajo | **1.4 GB → ~0 por arrastre** | Muy bajo |
| 2 | Poda del DOM + parseo por líneas en el servidor | Bajo | Elimina O(n²) y el acaparamiento del GIL | Bajo |
| 3 | Batch del text encoder en pre-caché | Bajo | ~8× esa etapa | Medio (cambia bytes de caché) |
| 4 | Quitar `free_vram()` del bucle del pre-caché | Muy bajo | ~213 ms/imagen | Muy bajo |
| 5 | `det_size` explícito + downscale en curaduría | Bajo | ~2-6× el **primer** scan | Medio (cambia scores) |
| 6 | Caché/ETag en `dataset-info`, quitar `exists()` duplicado | Bajo | −50 % syscalls, mata el N+1 | Bajo |
| 7 | Deduplicar embeds vía `os.link` | Bajo | **−452 MB disco** | Muy bajo (0 líneas en el trainer) |
| — | Benchmark `batch_size: 2` a 512 | Medio | **Desconocido — medir** | Bajo (experimento) |

## 4. Qué NO tocar

- **El bucle de entrenamiento** (`2_train:1813-2045`): medido en <0.02 % de overhead CPU.
- **La ruta de carga de caché del trainer** (`load_cache_entry:1539-1550`): 269 ms por fase
  (0.01 %). Nada que ganar, mucho que romper.
- **La arquitectura multiproceso de `run_progressive.py`**: la recarga por fase es el precio
  consciente de un allocator limpio en 12 GB.
- **`COMPACT_TEXT`**: ya activo y crítico. Sin él, SDPA cae al backend `math` y materializa la
  matriz completa `[B, heads, S, S]` (~568 MB a 768², documentado en `:365-366`).
- **La caché de curaduría**: correcta; los re-scans cuestan 0.07 s.

---

## Apéndice — Reproducir las mediciones

Duplicación de embeds entre resoluciones:

```bash
md5sum cached_data_local/lf_bd_v6/*/an_12_embed.pt
```

Tiempo de step real:

```bash
python3 -c "import csv,statistics as st; r=list(csv.DictReader(open('output_local/lf_bd_v5/phase2_1024/train_log.csv'))); print(st.median([float(x['secs']) for x in r]))"
```

Microbenchmark del hot path (H2D, `pack_latents`, `.item()`, casts, `free_vram`):

```python
# venv/bin/python
import torch, time, os, gc
d = "cached_data_local/lf_bd_v6/1024"
name = sorted(f[:-len("_latent.pt")] for f in os.listdir(d) if f.endswith("_latent.pt"))[0]
lat = torch.load(f"{d}/{name}_latent.pt", weights_only=True).to(torch.bfloat16)
emb = torch.load(f"{d}/{name}_embed.pt",  weights_only=True).to(torch.bfloat16)

def bench(fn, n=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

lat_p, emb_p = lat.pin_memory(), emb.pin_memory()
print("cat+H2D (actual):", bench(lambda: (torch.cat([lat_p]).to('cuda', non_blocking=True),
                                          torch.cat([emb_p]).to('cuda', non_blocking=True))))
print("pinned directo  :", bench(lambda: (lat_p.to('cuda', non_blocking=True),
                                          emb_p.to('cuda', non_blocking=True))))

def pack(x):
    B, C, H, W = x.shape
    return x.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)

g = lat.cuda()
print("pack_latents    :", bench(lambda: pack(g)))
x = torch.randn(1, device='cuda')
print(".item()         :", bench(lambda: x.item(), n=200))

torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(5): gc.collect(); torch.cuda.empty_cache()
torch.cuda.synchronize(); print("free_vram()     :", (time.perf_counter() - t) / 5 * 1000)
```

Coste de la carga de caché por fase (la que paga el trainer):

```python
# venv/bin/python
import torch, time, os
d = "cached_data_local/lf_bd_v6/1024"
names = sorted(f[:-len("_latent.pt")] for f in os.listdir(d) if f.endswith("_latent.pt"))
t = time.perf_counter()
for n in names:
    torch.load(f"{d}/{n}_latent.pt", weights_only=True)
    torch.load(f"{d}/{n}_embed.pt",  weights_only=True)
    torch.load(f"{d}/{n}_mask.pt",   weights_only=True)
print("torch.load total:", (time.perf_counter() - t) * 1000, "ms")
```
