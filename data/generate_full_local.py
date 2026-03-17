#!/usr/bin/env python3
"""
Generate 50K synthetic reasoning examples locally in batches of 1,000.
Extends the sample generator with more template variations for diversity.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate_synthetic import es_valido, format_example

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE = SCRIPT_DIR / "raw" / "phase2" / "synthetic_sample.jsonl"
OUTPUT = SCRIPT_DIR / "raw" / "phase2" / "synthetic.jsonl"

# =====================================================================
# ITEMS / PARAMETERS — large pools for combinatorial diversity
# =====================================================================

ITEMS_ES = [
    "libros", "camisetas", "bolígrafos", "cuadernos", "botellas de agua",
    "paquetes de galletas", "entradas de cine", "kilos de manzanas",
    "metros de tela", "cajas de cereal", "pares de calcetines",
    "barras de chocolate", "litros de leche", "kilos de arroz",
    "paquetes de café", "rollos de papel", "cajas de huevos",
    "kilos de tomates", "paquetes de pasta", "latas de atún",
    "botellas de jugo", "kilos de pollo", "bolsas de pan",
    "paquetes de servilletas", "kilos de queso", "litros de aceite",
    "cajas de té", "kilos de naranjas", "paquetes de mantequilla",
    "barras de jabón", "tubos de pasta dental", "kilos de papas",
]

ITEMS_EN = [
    "notebooks", "pens", "shirts", "water bottles", "bags of rice",
    "movie tickets", "pairs of socks", "boxes of cereal", "pencils",
    "loaves of bread", "bottles of juice", "cans of soup",
    "bags of flour", "bars of chocolate", "rolls of tape",
    "packs of gum", "boxes of tea", "cartons of milk",
    "pounds of chicken", "jars of honey", "bags of coffee",
    "tubes of toothpaste", "bars of soap", "packs of batteries",
    "reams of paper", "gallons of paint", "yards of fabric",
    "pounds of apples", "cans of tuna", "boxes of pasta",
    "pounds of cheese", "bottles of shampoo",
]

DISCOUNT_ITEMS_ES = [
    "una camisa", "un pantalón", "un par de zapatos", "una mochila",
    "un reloj", "una lámpara", "un libro de texto", "una silla",
    "una chaqueta", "un vestido", "una corbata", "un cinturón",
    "un sombrero", "una bufanda", "un abrigo", "una falda",
    "un suéter", "una blusa", "un maletín", "unas gafas de sol",
]

DISCOUNT_ITEMS_EN = [
    "a jacket", "a laptop bag", "a pair of shoes", "a watch",
    "a backpack", "a dress", "a belt", "a hat", "a scarf",
    "a coat", "a skirt", "a sweater", "a blouse", "a briefcase",
    "a pair of sunglasses", "a tie", "a lamp", "a chair",
    "a textbook", "a pair of gloves",
]

VEHICLES_ES = [
    "un auto", "un tren", "un autobús", "una bicicleta", "una moto",
    "un camión", "un barco", "un avión ligero", "una ambulancia",
    "un taxi", "una furgoneta", "un tranvía",
]

VEHICLES_EN = [
    "a car", "a train", "a bus", "a cyclist", "a motorcycle",
    "a truck", "a boat", "a light aircraft", "an ambulance",
    "a taxi", "a van", "a tram",
]

COVERAGE_ES = [
    "litros de pintura", "kilos de harina", "metros de cable",
    "litros de gasolina", "kilos de cemento", "rollos de papel tapiz",
    "kilos de fertilizante", "litros de barniz", "bolsas de tierra",
    "litros de impermeabilizante",
]

COVERAGE_EN = [
    "liters of paint", "bags of fertilizer", "rolls of wallpaper",
    "boxes of tiles", "bags of cement", "gallons of varnish",
    "rolls of insulation", "bags of gravel", "cans of sealant",
    "buckets of plaster",
]

PCT_CONTEXTS_ES = [
    ("una clase", "estudiantes", "aprobaron el examen"),
    ("una empresa", "empleados", "trabajan de forma remota"),
    ("una ciudad", "habitantes", "usan transporte público"),
    ("un grupo", "personas", "hablan más de un idioma"),
    ("una encuesta", "encuestados", "prefieren comprar en línea"),
    ("un hospital", "pacientes", "se recuperaron completamente"),
    ("un equipo", "jugadores", "anotaron al menos un gol"),
    ("una escuela", "alumnos", "obtuvieron calificación sobresaliente"),
    ("una fábrica", "trabajadores", "recibieron capacitación"),
    ("un país", "ciudadanos", "tienen acceso a internet"),
]

PCT_CONTEXTS_EN = [
    ("a school", "students", "passed the exam"),
    ("a company", "employees", "work remotely"),
    ("a survey", "respondents", "prefer online shopping"),
    ("a city", "residents", "use public transit"),
    ("a hospital", "patients", "recovered fully"),
    ("a team", "players", "scored at least one goal"),
    ("a class", "pupils", "achieved top marks"),
    ("a factory", "workers", "received training"),
    ("a country", "citizens", "have internet access"),
    ("a gym", "members", "attend more than twice a week"),
]

# =====================================================================
# SPANISH GENERATORS
# =====================================================================

def gen_es_compra(seed):
    random.seed(seed)
    item = random.choice(ITEMS_ES)
    precio = random.randint(2, 99)
    cantidad = random.randint(2, 15)
    total = precio * cantidad
    return {
        "pregunta": f"Si compras {cantidad} {item} a ${precio} cada uno, ¿cuánto pagas en total?",
        "razonamiento": [
            f"Paso 1: Sabemos que cada unidad de {item} tiene un precio de ${precio} según el enunciado.",
            f"Paso 2: El problema nos pide calcular el costo total para {cantidad} unidades.",
            f"Paso 3: Multiplicamos {cantidad} × {precio} = {total}, ya que el total es cantidad por precio unitario.",
        ],
        "conclusion": f"El total a pagar es ${total}.",
        "verificacion": f"{cantidad} * {precio} == {total}",
    }

def gen_es_descuento(seed):
    random.seed(seed)
    item = random.choice(DISCOUNT_ITEMS_ES)
    precio = random.choice([30, 40, 45, 50, 60, 75, 80, 90, 100, 120, 150, 200, 250])
    desc = random.choice([10, 15, 20, 25, 30, 40, 50])
    cantidad = random.randint(1, 6)
    ahorro = precio * desc // 100
    final = precio - ahorro
    total = final * cantidad
    return {
        "pregunta": f"Una tienda ofrece {desc}% de descuento en {item} de ${precio}. Si compras {cantidad}, ¿cuánto pagas?",
        "razonamiento": [
            f"Paso 1: El precio original de {item} es ${precio} según el enunciado del problema.",
            f"Paso 2: El descuento del {desc}% sobre ${precio} es {precio} × {desc}/100 = ${ahorro}, ya que multiplicamos por la tasa de descuento.",
            f"Paso 3: El precio con descuento es {precio} - {ahorro} = ${final}, porque restamos el ahorro del precio original.",
            f"Paso 4: Para {cantidad} unidades, el total es {final} × {cantidad} = ${total} según la multiplicación.",
        ],
        "conclusion": f"El total a pagar es ${total}.",
        "verificacion": f"({precio} - {precio} * {desc} // 100) * {cantidad} == {total}",
    }

def gen_es_velocidad(seed):
    random.seed(seed)
    vehiculo = random.choice(VEHICLES_ES)
    vel = random.choice([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    horas = random.choice([2, 3, 4, 5, 6, 8])
    dist = vel * horas
    return {
        "pregunta": f"Si {vehiculo} viaja a {vel} km/h durante {horas} horas, ¿qué distancia recorre?",
        "razonamiento": [
            f"Paso 1: Sabemos que la velocidad de {vehiculo} es de {vel} kilómetros por hora según el enunciado.",
            f"Paso 2: El tiempo de viaje es de {horas} horas, dato que nos proporciona el problema.",
            f"Paso 3: La distancia es velocidad × tiempo = {vel} × {horas} = {dist} km, ya que aplicamos la fórmula de movimiento uniforme.",
        ],
        "conclusion": f"La distancia recorrida es {dist} km.",
        "verificacion": f"{vel} * {horas} == {dist}",
    }

def gen_es_proporcion(seed):
    random.seed(seed)
    item = random.choice(COVERAGE_ES)
    cant1 = random.randint(2, 6)
    resultado1 = random.randint(8, 50)
    mult = random.randint(2, 5)
    cant2 = cant1 * mult
    resultado2 = resultado1 * mult
    return {
        "pregunta": f"Si {cant1} {item} alcanzan para {resultado1} m², ¿cuántos m² se cubren con {cant2} {item}?",
        "razonamiento": [
            f"Paso 1: Sabemos que {cant1} {item} cubren {resultado1} m² según el enunciado del problema.",
            f"Paso 2: Calculamos el rendimiento por unidad: {resultado1} ÷ {cant1} = {resultado1/cant1:.1f} m², ya que dividimos total entre cantidad.",
            f"Paso 3: Para {cant2} unidades, multiplicamos {resultado1/cant1:.1f} × {cant2} = {resultado2} m², porque el rendimiento es proporcional.",
        ],
        "conclusion": f"Con {cant2} {item} se cubren {resultado2} m².",
        "verificacion": f"{resultado1} * {cant2} // {cant1} == {resultado2}",
    }

def gen_es_porcentaje(seed):
    random.seed(seed)
    ctx = random.choice(PCT_CONTEXTS_ES)
    total = random.choice([20, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500])
    pct = random.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
    cantidad = total * pct // 100
    return {
        "pregunta": f"En {ctx[0]} de {total} {ctx[1]}, el {pct}% {ctx[2]}. ¿Cuántos son?",
        "razonamiento": [
            f"Paso 1: El total de {ctx[1]} es {total} según el enunciado del problema.",
            f"Paso 2: El porcentaje que {ctx[2]} es {pct}%, dato que nos proporciona el problema.",
            f"Paso 3: Calculamos {total} × {pct}/100 = {cantidad}, ya que multiplicamos el total por la fracción decimal.",
        ],
        "conclusion": f"{cantidad} {ctx[1]} {ctx[2]}.",
        "verificacion": f"{total} * {pct} // 100 == {cantidad}",
    }

def gen_es_dos_tramos(seed):
    random.seed(seed)
    v1 = random.choice([40, 50, 60, 80])
    h1 = random.choice([2, 3, 4])
    v2 = random.choice([30, 50, 70, 90, 100])
    h2 = random.choice([1, 2, 3])
    d1, d2 = v1*h1, v2*h2
    total = d1 + d2
    vehiculo = random.choice(VEHICLES_ES)
    return {
        "pregunta": f"Un viaje en {vehiculo}: {v1} km/h por {h1}h, luego {v2} km/h por {h2}h. ¿Distancia total?",
        "razonamiento": [
            f"Paso 1: En el primer tramo, {vehiculo} recorre {v1} × {h1} = {d1} km según la fórmula de distancia.",
            f"Paso 2: En el segundo tramo, recorre {v2} × {h2} = {d2} km aplicando la misma fórmula.",
            f"Paso 3: La distancia total es la suma de ambos tramos: {d1} + {d2} = {total} km, ya que sumamos las distancias parciales.",
        ],
        "conclusion": f"La distancia total recorrida es {total} km.",
        "verificacion": f"{v1} * {h1} + {v2} * {h2} == {total}",
    }

# --- Logic ES ---

_SYLLOGISMS_ES = [
    ("mamíferos", "vertebrados", "ballenas"), ("aves", "animales con plumas", "pingüinos"),
    ("insectos", "artrópodos", "hormigas"), ("reptiles", "animales de sangre fría", "lagartos"),
    ("árboles frutales", "plantas", "manzanos"), ("felinos", "carnívoros", "leones"),
    ("peces", "vertebrados acuáticos", "salmones"), ("anfibios", "vertebrados", "ranas"),
    ("crustáceos", "artrópodos", "cangrejos"), ("moluscos", "invertebrados", "pulpos"),
    ("cactus", "plantas suculentas", "nopales"), ("cítricos", "frutas", "limones"),
    ("roedores", "mamíferos", "ratones"), ("primates", "mamíferos", "gorilas"),
    ("coníferas", "árboles", "pinos"), ("legumbres", "alimentos vegetales", "lentejas"),
]

_MODUS_PONENS_ES = [
    ("llueve", "el suelo se moja", "está lloviendo", "el suelo está mojado"),
    ("hay fuego", "hay humo", "hay fuego en la cocina", "hay humo en la cocina"),
    ("es de día", "hay luz solar", "es mediodía", "hay luz solar"),
    ("la temperatura baja de 0°C", "el agua se congela", "la temperatura es -5°C", "el agua se congela"),
    ("un metal se calienta", "se dilata", "el hierro se calentó", "el hierro se dilató"),
    ("no hay oxígeno", "el fuego se apaga", "se selló la habitación", "el fuego se apagó"),
    ("un alumno estudia", "aprueba el examen", "Ana estudió toda la semana", "Ana aprobó el examen"),
    ("se planta una semilla con agua y sol", "germina", "se plantó con riego", "la semilla germinó"),
    ("hay exceso de oferta", "los precios bajan", "hubo sobreproducción", "los precios bajaron"),
    ("se corta la electricidad", "los aparatos se apagan", "hubo un apagón", "los aparatos se apagaron"),
]

_MODUS_TOLLENS_ES = [
    ("un animal es mamífero", "tiene sangre caliente", "no tiene sangre caliente", "no es mamífero"),
    ("una figura es un cuadrado", "tiene 4 lados iguales", "no tiene 4 lados iguales", "no es un cuadrado"),
    ("un número es par", "es divisible por 2", "no es divisible por 2", "no es par"),
    ("una sustancia es metal", "conduce electricidad", "no conduce electricidad", "no es un metal"),
    ("un triángulo es equilátero", "tiene 3 lados iguales", "no tiene 3 lados iguales", "no es equilátero"),
    ("un planeta es habitable", "tiene agua líquida", "no tiene agua líquida", "no es habitable"),
]

_DISYUNCION_ES = [
    ("va al cine", "va al teatro", "no fue al cine", "fue al teatro"),
    ("estudia medicina", "estudia derecho", "no estudia medicina", "estudia derecho"),
    ("toma el tren", "toma el autobús", "no tomó el tren", "tomó el autobús"),
    ("come carne", "come pescado", "no come carne", "come pescado"),
    ("viaja a París", "viaja a Londres", "no viajó a París", "viajó a Londres"),
    ("lee ficción", "lee ensayo", "no lee ficción", "lee ensayo"),
    ("juega fútbol", "juega tenis", "no juega fútbol", "juega tenis"),
    ("compra café", "compra té", "no compró café", "compró té"),
]

_TRANSITIVIDAD_ES = [
    ("A es mayor que B", "B es mayor que C", "A es mayor que C"),
    ("Pedro es más alto que Juan", "Juan es más alto que Luis", "Pedro es más alto que Luis"),
    ("el hierro es más denso que el aluminio", "el aluminio es más denso que la madera", "el hierro es más denso que la madera"),
    ("la empresa X factura más que Y", "Y factura más que Z", "X factura más que Z"),
    ("el río Amazonas es más largo que el Nilo", "el Nilo es más largo que el Misisipi", "el Amazonas es más largo que el Misisipi"),
    ("la temperatura de A es mayor que B", "B es mayor que C", "A tiene mayor temperatura que C"),
    ("el edificio A es más alto que B", "B es más alto que C", "el edificio A es más alto que C"),
]

def gen_es_logica(seed):
    random.seed(seed)
    tipo = random.randint(0, 4)
    if tipo == 0:
        A, B, C = random.choice(_SYLLOGISMS_ES)
        return {
            "pregunta": f"Si todos los {A} son {B}, y los {C} son {A}, ¿los {C} son {B}?",
            "razonamiento": [
                f"Paso 1: La primera premisa establece que todos los {A} pertenecen a la categoría de {B}.",
                f"Paso 2: La segunda premisa nos dice que los {C} son {A}, es decir, pertenecen a ese grupo.",
                f"Paso 3: Aplicando silogismo categórico, los {C} son {B} porque son {A} y todos los {A} son {B}.",
            ],
            "conclusion": f"Sí, los {C} son {B} por transitividad lógica.",
        }
    elif tipo == 1:
        p, q, hp, hq = random.choice(_MODUS_PONENS_ES)
        return {
            "pregunta": f"Si {p}, entonces {q}. Sabemos que {hp}. ¿Qué concluimos?",
            "razonamiento": [
                f"Paso 1: La premisa condicional establece que si {p}, entonces {q} como regla general.",
                f"Paso 2: Sabemos que {hp}, lo cual satisface la condición del antecedente según el enunciado.",
                f"Paso 3: Aplicando modus ponens, dado que el antecedente es verdadero, el consecuente también lo es.",
            ],
            "conclusion": f"Podemos concluir que {hq}.",
        }
    elif tipo == 2:
        p, q, nq, np_ = random.choice(_MODUS_TOLLENS_ES)
        return {
            "pregunta": f"Si {p}, entonces {q}. Sabemos que {nq}. ¿Qué concluimos?",
            "razonamiento": [
                f"Paso 1: La premisa condicional establece que si {p}, entonces necesariamente {q}.",
                f"Paso 2: Observamos que {nq}, lo cual niega el consecuente de la premisa condicional.",
                f"Paso 3: Aplicando modus tollens, dado que el consecuente es falso, el antecedente también lo es.",
            ],
            "conclusion": f"Podemos concluir que {np_}.",
        }
    elif tipo == 3:
        a, b, na, cb = random.choice(_DISYUNCION_ES)
        nombre = random.choice(["María", "Carlos", "Ana", "Pedro", "Laura", "Diego"])
        return {
            "pregunta": f"{nombre} {a} o {b}. Sabemos que {na}. ¿Qué hizo?",
            "razonamiento": [
                f"Paso 1: La premisa establece una disyunción: {nombre} {a} o {b} según el enunciado.",
                f"Paso 2: Sabemos que {nombre} {na}, lo cual elimina la primera opción de la disyunción.",
                f"Paso 3: Dado que una de las dos opciones debe ser verdadera y la primera es falsa, entonces {nombre} {cb}.",
            ],
            "conclusion": f"{nombre} {cb}.",
        }
    else:
        p1, p2, concl = random.choice(_TRANSITIVIDAD_ES)
        return {
            "pregunta": f"Si {p1} y {p2}, ¿qué relación hay entre el primero y el último?",
            "razonamiento": [
                f"Paso 1: La primera premisa establece que {p1} como dato del problema.",
                f"Paso 2: La segunda premisa nos dice que {p2}, lo cual conecta el segundo con el tercer elemento.",
                f"Paso 3: Aplicando la propiedad transitiva de la relación de orden, concluimos que {concl}.",
            ],
            "conclusion": f"{concl} por transitividad.",
        }

# --- Causal ES ---

_CAUSAL_CHAINS_ES = [
    ("se talan los bosques de una región",
     "Esto provoca que el suelo pierde la protección de las raíces contra la erosión.",
     "En consecuencia, las lluvias arrastran la capa fértil del suelo hacia los ríos.",
     "Esto causa que los ríos se llenen de sedimentos, reduciendo la calidad del agua.",
     "La deforestación degrada tanto el suelo como las fuentes de agua."),
    ("se introduce una especie invasora en un lago",
     "Esto significa que los peces nativos enfrentan competencia directa por alimento y espacio.",
     "Dado que la especie invasora se reproduce más rápido, desplaza gradualmente a las nativas.",
     "En consecuencia, los depredadores que dependían de peces nativos pierden su fuente de alimento.",
     "La especie invasora altera toda la cadena alimenticia del lago."),
    ("aumenta la temperatura global en 2°C",
     "Esto provoca que los glaciares polares se derriten a un ritmo acelerado.",
     "En consecuencia, el nivel del mar sube e inunda zonas costeras de todo el mundo.",
     "Esto causa que millones de personas deban migrar hacia zonas más elevadas.",
     "El calentamiento global desencadena una crisis migratoria y ambiental."),
    ("el banco central sube las tasas de interés",
     "Esto significa que los préstamos se vuelven más caros para empresas y consumidores.",
     "En consecuencia, las personas gastan menos y las empresas reducen inversiones.",
     "Esto causa que la demanda agregada disminuya, lo cual frena el crecimiento económico.",
     "La subida de tasas reduce la inflación pero desacelera la economía."),
    ("un país devalúa su moneda",
     "Esto implica que los productos importados se vuelven más caros para consumidores locales.",
     "Como resultado, las exportaciones se vuelven más competitivas en mercados internacionales.",
     "En consecuencia, la balanza comercial mejora pero el poder adquisitivo interno se reduce.",
     "La devaluación beneficia a exportadores pero perjudica a consumidores."),
    ("aumenta el precio del petróleo un 50%",
     "Esto provoca que los costos de transporte y logística se incrementan significativamente.",
     "Como resultado, los precios de todos los productos que requieren transporte también suben.",
     "Esto causa que la inflación general se acelere y el poder de compra disminuya.",
     "El encarecimiento del petróleo genera una ola inflacionaria."),
    ("una persona no duerme lo suficiente durante semanas",
     "Sabemos que la falta de sueño reduce la producción de hormonas del sistema inmune.",
     "Esto significa que el cuerpo pierde capacidad para combatir infecciones comunes.",
     "En consecuencia, la persona enferma con más frecuencia y su recuperación es más lenta.",
     "La privación crónica de sueño debilita el sistema inmunológico."),
    ("una persona deja de hacer ejercicio por meses",
     "Sabemos que la inactividad prolongada reduce la masa muscular y la capacidad cardiovascular.",
     "Esto provoca que el metabolismo basal disminuya y el cuerpo queme menos calorías.",
     "Como resultado, la persona acumula grasa corporal y pierde resistencia física.",
     "El sedentarismo deteriora la composición corporal y la salud cardiovascular."),
    ("se implementa educación gratuita universitaria",
     "Esto significa que las barreras económicas para acceder a la universidad se eliminan.",
     "En consecuencia, más jóvenes de familias de bajos ingresos obtienen títulos profesionales.",
     "Esto causa que la movilidad social aumente y la desigualdad se reduzca a largo plazo.",
     "La educación gratuita aumenta la movilidad social y reduce la desigualdad."),
    ("una ciudad prohíbe los autos en su centro histórico",
     "Esto implica que los residentes y visitantes deben usar transporte público o caminar.",
     "Como resultado, la contaminación del aire en el centro disminuye significativamente.",
     "En consecuencia, la calidad de vida mejora y el turismo peatonal aumenta.",
     "La peatonalización reduce la contaminación y mejora la experiencia urbana."),
    ("se calienta un gas en un recipiente cerrado",
     "Sabemos que al aumentar la temperatura las moléculas se mueven con mayor velocidad.",
     "Esto significa que las moléculas chocan más frecuentemente contra las paredes del recipiente.",
     "En consecuencia, la presión del gas dentro del recipiente aumenta proporcionalmente.",
     "Calentar un gas en recipiente cerrado aumenta su presión, según Gay-Lussac."),
    ("se aplica una fuerza constante a un objeto sin fricción",
     "Sabemos que la segunda ley de Newton establece que fuerza es igual a masa por aceleración.",
     "Esto implica que el objeto experimenta una aceleración constante en la dirección de la fuerza.",
     "Como resultado, la velocidad del objeto aumenta de forma lineal con el tiempo.",
     "El objeto acelera uniformemente mientras se mantenga la fuerza aplicada."),
    ("se reduce drásticamente la emisión de CO2 global",
     "Esto significa que la concentración de gases de efecto invernadero en la atmósfera se estabiliza.",
     "En consecuencia, el ritmo de calentamiento global se desacelera gradualmente.",
     "Esto causa que los ecosistemas tengan más tiempo para adaptarse a los cambios climáticos.",
     "La reducción de emisiones frena el calentamiento y da tiempo a la adaptación ecológica."),
    ("un volcán entra en erupción cerca de una ciudad",
     "Esto provoca que grandes cantidades de ceniza y gases tóxicos se liberen a la atmósfera.",
     "Como resultado, la calidad del aire se deteriora severamente en toda la región cercana.",
     "En consecuencia, las autoridades deben evacuar a la población por riesgo de salud.",
     "La erupción volcánica contamina el aire y obliga a evacuaciones masivas."),
]

def gen_es_causal(seed):
    random.seed(seed)
    preg, p1, p2, p3, concl = random.choice(_CAUSAL_CHAINS_ES)
    return {
        "pregunta": f"¿Qué consecuencias tiene que {preg}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }

# --- Analogies ES ---

_ANALOGIES_ES = [
    ("el sistema inmunológico y un ejército",
     "La piel funciona como las murallas de una fortaleza, ya que ambas bloquean la entrada de invasores.",
     "Los glóbulos blancos actúan como soldados de patrulla, porque detectan y atacan agentes extraños.",
     "Los anticuerpos son como órdenes de captura específicas, dado que cada uno reconoce un enemigo particular.",
     "Ambos sistemas comparten barrera, fuerza activa y respuesta específica."),
    ("el cerebro humano y una computadora",
     "Las neuronas funcionan como transistores, ya que ambas procesan señales eléctricas.",
     "La memoria a corto plazo es análoga a la RAM, porque ambas almacenan información temporal.",
     "El aprendizaje equivale a actualizar software, dado que ambos modifican conexiones o instrucciones.",
     "Ambos procesan información, almacenan datos y se adaptan."),
    ("una empresa y un organismo vivo",
     "El departamento financiero funciona como el sistema circulatorio, ya que ambos distribuyen recursos.",
     "Los empleados son como las células, porque cada uno cumple una función especializada.",
     "La dirección actúa como el sistema nervioso central, dado que coordina y toma decisiones.",
     "Ambos tienen componentes especializados, distribución de recursos y coordinación central."),
    ("el ciclo del agua y el ciclo económico del dinero",
     "La evaporación es como el ahorro, ya que en ambos los recursos se retiran de circulación.",
     "La precipitación equivale al gasto público, porque en ambos los recursos vuelven a circular.",
     "Los ríos funcionan como los bancos, dado que ambos canalizan y distribuyen recursos.",
     "Ambos ciclos muestran acumulación, liberación y distribución de un recurso limitado."),
    ("una biblioteca y un motor de búsqueda web",
     "El catálogo funciona como el índice del buscador, ya que ambos organizan referencias al contenido.",
     "Los bibliotecarios actúan como el algoritmo de ranking, porque ambos evalúan la relevancia.",
     "Las estanterías son como los servidores, dado que ambos almacenan físicamente la información.",
     "Ambos organizan, clasifican y facilitan el acceso a grandes volúmenes de información."),
    ("el ADN y un programa informático",
     "Los genes funcionan como funciones del código, ya que cada uno contiene instrucciones específicas.",
     "Las mutaciones son análogas a los bugs, porque ambas son cambios que alteran el resultado.",
     "La transcripción equivale a la compilación, dado que ambos convierten instrucciones en productos.",
     "Ambos codifican instrucciones, pueden tener errores y requieren traducción para funcionar."),
    ("un ecosistema y una economía de mercado",
     "Las especies productoras son como las empresas, ya que ambas generan los recursos base del sistema.",
     "Los depredadores funcionan como los reguladores del mercado, porque controlan el exceso de producción.",
     "La extinción de una especie es como la quiebra de una empresa clave, dado que ambas desestabilizan el sistema.",
     "Ambos dependen del equilibrio entre producción, consumo y regulación."),
    ("el sistema solar y un átomo",
     "El sol funciona como el núcleo atómico, ya que ambos son el centro masivo del sistema.",
     "Los planetas orbitan como los electrones, porque ambos siguen trayectorias alrededor del centro.",
     "La gravedad actúa como la fuerza electromagnética, dado que ambas mantienen los cuerpos en órbita.",
     "Ambos sistemas tienen un centro masivo con cuerpos orbitando a distintas distancias."),
]

def gen_es_analogia(seed):
    random.seed(seed)
    tema, p1, p2, p3, concl = random.choice(_ANALOGIES_ES)
    return {
        "pregunta": f"¿En qué se parecen {tema}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }

# =====================================================================
# ENGLISH GENERATORS
# =====================================================================

def gen_en_shopping(seed):
    random.seed(seed)
    item = random.choice(ITEMS_EN)
    price = random.randint(2, 99)
    qty = random.randint(2, 15)
    total = price * qty
    return {
        "pregunta": f"You buy {qty} {item} at ${price} each. How much do you spend?",
        "razonamiento": [
            f"Step 1: The problem states that each unit of {item} costs ${price} at the store.",
            f"Step 2: We need to calculate the total cost for {qty} units according to the problem.",
            f"Step 3: We calculate {qty} × {price} = {total}, since total cost equals quantity times unit price.",
        ],
        "conclusion": f"The total cost is ${total}.",
        "verificacion": f"{qty} * {price} == {total}",
    }

def gen_en_discount(seed):
    random.seed(seed)
    item = random.choice(DISCOUNT_ITEMS_EN)
    price = random.choice([30, 40, 50, 60, 75, 80, 100, 120, 150, 200])
    disc = random.choice([10, 15, 20, 25, 30, 40, 50])
    qty = random.randint(1, 5)
    savings = price * disc // 100
    final = price - savings
    total = final * qty
    return {
        "pregunta": f"A store offers {disc}% off on {item} priced at ${price}. You buy {qty}. How much?",
        "razonamiento": [
            f"Step 1: The original price of {item} is ${price} according to the problem statement.",
            f"Step 2: The {disc}% discount means we save {price} × {disc}/100 = ${savings}, since we multiply by the rate.",
            f"Step 3: The discounted price is {price} - {savings} = ${final}, because we subtract savings from the original.",
            f"Step 4: For {qty} units the total is {final} × {qty} = ${total}, since we multiply unit price by quantity.",
        ],
        "conclusion": f"The total cost is ${total}.",
        "verificacion": f"({price} - {price} * {disc} // 100) * {qty} == {total}",
    }

def gen_en_distance(seed):
    random.seed(seed)
    vehicle = random.choice(VEHICLES_EN)
    speed = random.choice([30, 40, 50, 60, 80, 100, 120])
    hours = random.choice([2, 3, 4, 5, 6])
    dist = speed * hours
    return {
        "pregunta": f"If {vehicle} travels at {speed} km/h for {hours} hours, what distance does it cover?",
        "razonamiento": [
            f"Step 1: We know that {vehicle} maintains a speed of {speed} km/h according to the problem.",
            f"Step 2: The travel time is {hours} hours, which is given in the problem statement.",
            f"Step 3: We calculate distance = speed × time = {speed} × {hours} = {dist} km, since distance equals rate times time.",
        ],
        "conclusion": f"The distance covered is {dist} km.",
        "verificacion": f"{speed} * {hours} == {dist}",
    }

def gen_en_percentage(seed):
    random.seed(seed)
    ctx = random.choice(PCT_CONTEXTS_EN)
    total = random.choice([20, 40, 50, 80, 100, 120, 200, 250, 400, 500])
    pct = random.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
    result = total * pct // 100
    return {
        "pregunta": f"In {ctx[0]} with {total} {ctx[1]}, {pct}% {ctx[2]}. How many?",
        "razonamiento": [
            f"Step 1: The total number of {ctx[1]} is {total} according to the problem statement.",
            f"Step 2: We need to find {pct}% of {total}, which means multiplying by the decimal equivalent.",
            f"Step 3: We calculate {total} × {pct}/100 = {result}, since percentage means parts per hundred.",
        ],
        "conclusion": f"{result} {ctx[1]} {ctx[2]}.",
        "verificacion": f"{total} * {pct} // 100 == {result}",
    }

def gen_en_ratio(seed):
    random.seed(seed)
    item = random.choice(COVERAGE_EN)
    a1 = random.randint(2, 6)
    c1 = random.randint(8, 50)
    mult = random.randint(2, 5)
    a2 = a1 * mult
    c2 = c1 * mult
    return {
        "pregunta": f"If {a1} {item} cover {c1} m², how many m² can {a2} {item} cover?",
        "razonamiento": [
            f"Step 1: We know that {a1} {item} cover {c1} m² according to the problem statement.",
            f"Step 2: We calculate per-unit coverage: {c1} ÷ {a1} = {c1/a1:.1f} m², since we divide total by quantity.",
            f"Step 3: For {a2} units we multiply {c1/a1:.1f} × {a2} = {c2} m², because coverage scales proportionally.",
        ],
        "conclusion": f"{a2} {item} cover {c2} m².",
        "verificacion": f"{c1} * {a2} // {a1} == {c2}",
    }

def gen_en_two_legs(seed):
    random.seed(seed)
    v1 = random.choice([40, 50, 60, 80])
    h1 = random.choice([2, 3, 4])
    v2 = random.choice([30, 50, 70, 90, 100])
    h2 = random.choice([1, 2, 3])
    d1, d2 = v1*h1, v2*h2
    total = d1 + d2
    vehicle = random.choice(VEHICLES_EN)
    return {
        "pregunta": f"A trip by {vehicle}: {v1} km/h for {h1}h, then {v2} km/h for {h2}h. Total distance?",
        "razonamiento": [
            f"Step 1: In the first leg, {vehicle} covers {v1} × {h1} = {d1} km according to the distance formula.",
            f"Step 2: In the second leg, it covers {v2} × {h2} = {d2} km applying the same formula.",
            f"Step 3: The total distance is the sum of both legs: {d1} + {d2} = {total} km, since we add partial distances.",
        ],
        "conclusion": f"The total distance traveled is {total} km.",
        "verificacion": f"{v1} * {h1} + {v2} * {h2} == {total}",
    }

# --- Logic EN ---

_SYLLOGISMS_EN = [
    ("mammals", "vertebrates", "dolphins"), ("birds", "feathered animals", "eagles"),
    ("insects", "arthropods", "butterflies"), ("reptiles", "cold-blooded animals", "crocodiles"),
    ("fungi", "organisms without chlorophyll", "mushrooms"), ("fish", "aquatic vertebrates", "salmon"),
    ("amphibians", "vertebrates", "frogs"), ("crustaceans", "arthropods", "crabs"),
    ("mollusks", "invertebrates", "octopuses"), ("conifers", "trees", "pines"),
    ("rodents", "mammals", "mice"), ("primates", "mammals", "gorillas"),
    ("citrus fruits", "fruits", "lemons"), ("legumes", "plant foods", "lentils"),
]

_MODUS_PONENS_EN = [
    ("it rains", "the streets get wet", "it is raining", "the streets are wet"),
    ("a metal is heated", "it expands", "the iron bar was heated", "the iron bar expanded"),
    ("the sun sets", "it gets dark", "the sun has set", "it is dark outside"),
    ("you study hard", "you pass the exam", "Maria studied hard", "Maria passed the exam"),
    ("there is no oxygen", "the fire goes out", "the room was sealed", "the fire went out"),
    ("demand exceeds supply", "prices rise", "demand exceeded supply", "prices rose"),
    ("a seed gets water and sunlight", "it germinates", "the seed was watered", "it germinated"),
    ("the power goes out", "the devices shut off", "there was a blackout", "the devices shut off"),
]

_MODUS_TOLLENS_EN = [
    ("an animal is a fish", "it lives in water", "it does not live in water", "it is not a fish"),
    ("a shape is a circle", "it has no corners", "it has corners", "it is not a circle"),
    ("a number is even", "it is divisible by 2", "it is not divisible by 2", "it is not even"),
    ("a substance is a metal", "it conducts electricity", "it does not conduct electricity", "it is not a metal"),
    ("a planet is habitable", "it has liquid water", "it has no liquid water", "it is not habitable"),
    ("a triangle is equilateral", "all sides are equal", "the sides are not equal", "it is not equilateral"),
]

_DISYUNCION_EN = [
    ("takes the bus", "takes the train", "did not take the bus", "took the train"),
    ("studies physics", "studies chemistry", "does not study physics", "studies chemistry"),
    ("goes to the park", "goes to the museum", "did not go to the park", "went to the museum"),
    ("eats chicken", "eats fish", "did not eat chicken", "ate fish"),
    ("travels to Paris", "travels to London", "did not travel to Paris", "traveled to London"),
    ("reads fiction", "reads nonfiction", "does not read fiction", "reads nonfiction"),
    ("plays soccer", "plays tennis", "does not play soccer", "plays tennis"),
]

_TRANSITIVIDAD_EN = [
    ("A is taller than B", "B is taller than C", "A is taller than C"),
    ("iron is denser than aluminum", "aluminum is denser than wood", "iron is denser than wood"),
    ("Tokyo is larger than London", "London is larger than Paris", "Tokyo is larger than Paris"),
    ("the Amazon is longer than the Nile", "the Nile is longer than the Mississippi", "the Amazon is longer than the Mississippi"),
    ("building A is taller than B", "B is taller than C", "building A is taller than C"),
]

def gen_en_logic(seed):
    random.seed(seed)
    tipo = random.randint(0, 4)
    if tipo == 0:
        A, B, C = random.choice(_SYLLOGISMS_EN)
        return {
            "pregunta": f"All {A} are {B}. {C} are {A}. Are {C} also {B}?",
            "razonamiento": [
                f"Step 1: The first premise establishes that all {A} belong to the category of {B}.",
                f"Step 2: The second premise tells us that {C} are {A}, meaning they belong to that group.",
                f"Step 3: Applying categorical syllogism, {C} are {B} because they are {A} and all {A} are {B}.",
            ],
            "conclusion": f"Yes, {C} are {B} by logical transitivity.",
        }
    elif tipo == 1:
        p, q, hp, hq = random.choice(_MODUS_PONENS_EN)
        return {
            "pregunta": f"If {p}, then {q}. We know that {hp}. What can we conclude?",
            "razonamiento": [
                f"Step 1: The conditional premise establishes that if {p}, then {q} as a general rule.",
                f"Step 2: We know that {hp}, which satisfies the antecedent according to the given information.",
                f"Step 3: Applying modus ponens, since the antecedent is true, the consequent must also be true.",
            ],
            "conclusion": f"We conclude that {hq}.",
        }
    elif tipo == 2:
        p, q, nq, np_ = random.choice(_MODUS_TOLLENS_EN)
        return {
            "pregunta": f"If {p}, then {q}. We observe that {nq}. What follows?",
            "razonamiento": [
                f"Step 1: The conditional premise states that if {p}, then necessarily {q}.",
                f"Step 2: We observe that {nq}, which negates the consequent of the conditional.",
                f"Step 3: Applying modus tollens, since the consequent is false, the antecedent must also be false.",
            ],
            "conclusion": f"We conclude that {np_}.",
        }
    elif tipo == 3:
        a, b, na, cb = random.choice(_DISYUNCION_EN)
        name = random.choice(["Tom", "Sarah", "Alex", "Emma", "James", "Lisa"])
        return {
            "pregunta": f"{name} {a} or {b}. We know {name} {na}. What did {name} do?",
            "razonamiento": [
                f"Step 1: The premise establishes a disjunction: {name} {a} or {b} according to the problem.",
                f"Step 2: We know that {name} {na}, which eliminates the first option from the disjunction.",
                f"Step 3: Since one option must be true and the first is false, {name} {cb} by elimination.",
            ],
            "conclusion": f"{name} {cb}.",
        }
    else:
        p1, p2, concl = random.choice(_TRANSITIVIDAD_EN)
        return {
            "pregunta": f"If {p1}, and {p2}, what is the relationship between the first and last?",
            "razonamiento": [
                f"Step 1: The first premise establishes that {p1} as a given fact in the problem.",
                f"Step 2: The second premise tells us that {p2}, connecting the second to the third element.",
                f"Step 3: Applying the transitive property of ordering, we conclude that {concl}.",
            ],
            "conclusion": f"{concl}, by transitivity.",
        }

# --- Causal EN ---

_CAUSAL_CHAINS_EN = [
    ("deforestation increases in a tropical region",
     "This causes the soil to lose the protective root systems that held it in place.",
     "As a result, heavy rains wash away the fertile topsoil into rivers and streams.",
     "Consequently, the rivers fill with sediment and water quality deteriorates significantly.",
     "Deforestation leads to soil erosion and degradation of water sources."),
    ("a city bans single-use plastics",
     "This means that businesses must switch to biodegradable or reusable alternatives.",
     "As a result, the volume of plastic waste entering landfills and oceans decreases.",
     "Consequently, marine ecosystems begin to recover as less pollution enters the food chain.",
     "The plastic ban reduces waste and helps restore marine ecosystems."),
    ("interest rates rise sharply",
     "This causes borrowing to become more expensive for consumers and businesses alike.",
     "As a result, consumer spending decreases and businesses delay investment projects.",
     "Consequently, economic growth slows down and unemployment may increase short-term.",
     "Rising interest rates slow growth by reducing spending and investment."),
    ("a vaccine is widely distributed",
     "This means that a large percentage of the population develops immunity.",
     "As a result, the transmission rate drops because fewer susceptible hosts are available.",
     "Consequently, disease incidence falls dramatically and herd immunity is achieved.",
     "Widespread vaccination leads to herd immunity and controls disease spread."),
    ("a major earthquake strikes near a coast",
     "This causes massive displacement of the ocean floor, generating powerful waves.",
     "As a result, a tsunami forms and travels at high speed toward the coastline.",
     "Consequently, coastal areas experience severe flooding and infrastructure damage.",
     "Coastal earthquakes can trigger tsunamis causing devastating flooding."),
    ("global CO2 emissions are drastically reduced",
     "This means that greenhouse gas concentration in the atmosphere stabilizes over time.",
     "As a result, the rate of global warming decelerates gradually over decades.",
     "Consequently, ecosystems have more time to adapt to the changing climate conditions.",
     "Emission reductions slow warming and give ecosystems time to adapt."),
    ("a volcano erupts near a populated area",
     "This causes large amounts of ash and toxic gases to be released into the atmosphere.",
     "As a result, air quality deteriorates severely across the entire surrounding region.",
     "Consequently, authorities must evacuate the population due to serious health risks.",
     "Volcanic eruptions contaminate the air and force mass evacuations."),
    ("a country invests heavily in renewable energy",
     "This means that the share of fossil fuels in electricity generation decreases steadily.",
     "As a result, carbon emissions from the energy sector drop significantly over time.",
     "Consequently, the country becomes less dependent on imported fossil fuels and more energy-secure.",
     "Renewable investment reduces emissions and increases energy independence."),
    ("ocean temperatures rise due to climate change",
     "This causes coral reefs to experience thermal stress and begin bleaching.",
     "As a result, the biodiversity supported by coral ecosystems declines sharply.",
     "Consequently, fishing communities that depend on reef fish lose their primary food source.",
     "Ocean warming destroys coral reefs and devastates dependent communities."),
    ("a pandemic forces widespread remote work adoption",
     "This means that companies invest in digital infrastructure and collaboration tools.",
     "As a result, employees discover that many jobs can be done effectively from home.",
     "Consequently, demand for commercial office space decreases permanently in many cities.",
     "The pandemic accelerated remote work and reduced demand for office space."),
]

def gen_en_causal(seed):
    random.seed(seed)
    topic, p1, p2, p3, concl = random.choice(_CAUSAL_CHAINS_EN)
    return {
        "pregunta": f"What happens when {topic}?",
        "razonamiento": [f"Step 1: {p1}", f"Step 2: {p2}", f"Step 3: {p3}"],
        "conclusion": concl,
    }

# --- Analogies EN ---

_ANALOGIES_EN = [
    ("the human heart and a water pump",
     "The heart chambers function like pump compartments, since both create pressure to move fluid.",
     "The valves in the heart work like check valves, because both prevent backflow of fluid.",
     "The arteries are like distribution pipes, given that both carry pressurized fluid to destinations.",
     "Both systems use chambers, valves, and channels to circulate fluid through a network."),
    ("a cell and a factory",
     "The nucleus acts like the management office, since both contain instructions guiding operations.",
     "The mitochondria function like power generators, because both convert materials into energy.",
     "The cell membrane works like the security perimeter, given that both control entry and exit.",
     "Both systems have central control, energy production, and controlled boundaries."),
    ("evolution and machine learning",
     "Genetic mutations are like random parameter changes, since both introduce variation.",
     "Natural selection works like the loss function, because both determine which variations survive.",
     "Successive generations are like training epochs, given that both iteratively improve performance.",
     "Both use random variation, selective pressure, and iteration to optimize solutions."),
    ("a democracy and a marketplace",
     "Voting functions like purchasing, since both let individuals express preferences through choices.",
     "Political parties act like competing brands, because both try to attract the largest support share.",
     "Elections work like market cycles, given that both periodically redistribute power based on preferences.",
     "Both aggregate individual preferences to determine collective outcomes through competition."),
    ("the internet and the postal system",
     "Data packets are like letters, since both carry information from sender to recipient.",
     "Routers function like sorting offices, because both direct items along efficient paths.",
     "IP addresses work like postal addresses, given that both uniquely identify delivery destinations.",
     "Both route addressed packages of information through a network of intermediate nodes."),
    ("DNA and a computer program",
     "Genes function like code functions, since each contains instructions for a specific task.",
     "Mutations are analogous to bugs, because both are changes that alter expected results.",
     "Transcription is like compilation, given that both convert instructions into executable products.",
     "Both encode instructions, can have errors, and require translation to function."),
    ("an ecosystem and a market economy",
     "Producer species are like businesses, since both generate the base resources of the system.",
     "Predators function like market regulators, because both control excess production and maintain balance.",
     "Species extinction is like a key company failing, given that both destabilize the entire system.",
     "Both depend on equilibrium between production, consumption, and regulation."),
    ("the solar system and an atom",
     "The sun functions like the atomic nucleus, since both are the massive center of the system.",
     "Planets orbit like electrons, because both follow trajectories around the central body.",
     "Gravity acts like the electromagnetic force, given that both keep bodies in stable orbits.",
     "Both have a massive center with bodies orbiting at various distances."),
]

def gen_en_analogy(seed):
    random.seed(seed)
    topic, p1, p2, p3, concl = random.choice(_ANALOGIES_EN)
    return {
        "pregunta": f"How are {topic} similar?",
        "razonamiento": [f"Step 1: {p1}", f"Step 2: {p2}", f"Step 3: {p3}"],
        "conclusion": concl,
    }


# =====================================================================
# MAIN — Generate in batches of 1,000
# =====================================================================

ALL_ES = [
    ("matematicas", gen_es_compra), ("matematicas", gen_es_descuento),
    ("matematicas", gen_es_velocidad), ("matematicas", gen_es_proporcion),
    ("matematicas", gen_es_porcentaje), ("matematicas", gen_es_dos_tramos),
    ("logica", gen_es_logica),
    ("causal", gen_es_causal),
    ("analogias", gen_es_analogia),
]

ALL_EN = [
    ("math", gen_en_shopping), ("math", gen_en_discount),
    ("math", gen_en_distance), ("math", gen_en_percentage),
    ("math", gen_en_ratio), ("math", gen_en_two_legs),
    ("logic", gen_en_logic),
    ("causal", gen_en_causal),
    ("analogies", gen_en_analogy),
]

def generate_one(lang, seed):
    """Generate one example for the given language."""
    if lang == "es":
        cat, gen_fn = random.choice(ALL_ES)
        ex = gen_fn(seed)
        return ex, cat, "es"
    else:
        cat, gen_fn = random.choice(ALL_EN)
        ex = gen_fn(seed)
        return ex, cat, "en"


def main():
    import shutil

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Start from sample if it exists
    if SAMPLE.exists():
        shutil.copy2(SAMPLE, OUTPUT)
        with open(SAMPLE) as f:
            existing = sum(1 for _ in f)
        print(f"Copied {existing} sample examples to {OUTPUT}")
    else:
        existing = 0
        open(OUTPUT, "w").close()

    TARGET = 50_000
    remaining = TARGET - existing
    BATCH = 1000

    total_accepted = existing
    total_rejected = 0
    rejected_reasons = {}
    seed_counter = existing * 10  # avoid seed overlap with sample

    batch_num = 0
    while total_accepted < TARGET:
        batch_num += 1
        batch_target = min(BATCH, TARGET - total_accepted)
        batch_es = int(batch_target * 0.7)
        batch_en = batch_target - batch_es

        batch_records = []
        batch_rejected = 0
        batch_reject_reasons = {}

        # Generate ES
        attempts = 0
        got = 0
        while got < batch_es and attempts < batch_es * 3:
            seed_counter += 1
            attempts += 1
            random.seed(seed_counter)
            ex, cat, lang = generate_one("es", seed_counter)
            ok, reason = es_valido(ex)
            if ok:
                text = format_example(ex, "es")
                record = {"text": text, "lang": "es", "category": cat}
                if "verificacion" in ex:
                    record["verificacion"] = ex["verificacion"]
                batch_records.append(record)
                got += 1
            else:
                batch_rejected += 1
                batch_reject_reasons[reason] = batch_reject_reasons.get(reason, 0) + 1

        # Generate EN
        attempts = 0
        got = 0
        while got < batch_en and attempts < batch_en * 3:
            seed_counter += 1
            attempts += 1
            random.seed(seed_counter)
            ex, cat, lang = generate_one("en", seed_counter)
            ok, reason = es_valido(ex)
            if ok:
                text = format_example(ex, "en")
                record = {"text": text, "lang": "en", "category": cat}
                if "verificacion" in ex:
                    record["verificacion"] = ex["verificacion"]
                batch_records.append(record)
                got += 1
            else:
                batch_rejected += 1
                batch_reject_reasons[reason] = batch_reject_reasons.get(reason, 0) + 1

        # Shuffle and write batch
        random.shuffle(batch_records)
        with open(OUTPUT, "a") as f:
            for r in batch_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_accepted += len(batch_records)
        total_rejected += batch_rejected
        for k, v in batch_reject_reasons.items():
            rejected_reasons[k] = rejected_reasons.get(k, 0) + v

        reject_rate = batch_rejected / (len(batch_records) + batch_rejected) * 100 if (len(batch_records) + batch_rejected) > 0 else 0
        print(f"Batch {batch_num:3d}: +{len(batch_records):4d} accepted, {batch_rejected:3d} rejected ({reject_rate:.1f}%)  |  Total: {total_accepted:,}/{TARGET:,}")

    # Final stats
    print(f"\n{'='*60}")
    print(f"DONE: {total_accepted:,} examples in {OUTPUT}")
    print(f"Total rejected: {total_rejected:,}")
    if rejected_reasons:
        print(f"Rejection reasons: {rejected_reasons}")

    # Count distribution
    es_count = en_count = 0
    cats = {}
    with open(OUTPUT) as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] == "es":
                es_count += 1
            else:
                en_count += 1
            cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"Distribution: {es_count:,} ES ({es_count/total_accepted*100:.0f}%) / {en_count:,} EN ({en_count/total_accepted*100:.0f}%)")
    print(f"Categories: {cats}")


if __name__ == "__main__":
    main()
