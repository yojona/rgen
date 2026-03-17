#!/usr/bin/env python3
"""
Generate 500 synthetic reasoning examples locally (no API needed).

Uses parameterized templates with random variations to produce
diverse examples across 4 categories × 2 languages.

Distribution: 250 ES + 250 EN (62-63 per category per language).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Add parent to path so we can import es_valido
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate_synthetic import es_valido, format_example

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "raw" / "phase2" / "synthetic_sample.jsonl"

random.seed(42)

# =====================================================================
# SPANISH TEMPLATES
# =====================================================================

def gen_es_matematicas(seed: int) -> dict:
    random.seed(seed)
    templates = [
        # Compras simples
        lambda: _es_compra_simple(),
        # Descuentos
        lambda: _es_descuento(),
        # Velocidad/distancia/tiempo
        lambda: _es_velocidad(),
        # Proporciones
        lambda: _es_proporcion(),
        # Porcentajes
        lambda: _es_porcentaje(),
    ]
    return random.choice(templates)()


def _es_compra_simple():
    item = random.choice(["libros", "camisetas", "bolígrafos", "cuadernos", "botellas de agua", "paquetes de galletas", "entradas de cine", "kilos de manzanas", "metros de tela", "cajas de cereal"])
    precio = random.randint(3, 50)
    cantidad = random.randint(2, 12)
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


def _es_descuento():
    item = random.choice(["una camisa", "un pantalón", "un par de zapatos", "una mochila", "un reloj", "una lámpara", "un libro de texto", "una silla"])
    precio = random.choice([40, 45, 50, 60, 75, 80, 90, 100, 120, 150])
    desc_pct = random.choice([10, 15, 20, 25, 30])
    cantidad = random.randint(1, 5)
    descuento = precio * desc_pct / 100
    precio_final = precio - descuento
    total = precio_final * cantidad
    return {
        "pregunta": f"Una tienda ofrece {desc_pct}% de descuento en {item} que cuesta ${precio}. Si compras {cantidad}, ¿cuánto pagas?",
        "razonamiento": [
            f"Paso 1: El precio original de {item} es ${precio} según el enunciado del problema.",
            f"Paso 2: El descuento del {desc_pct}% sobre ${precio} es {precio} × {desc_pct/100} = ${descuento:.0f}, ya que multiplicamos por la tasa.",
            f"Paso 3: El precio con descuento es {precio} - {descuento:.0f} = ${precio_final:.0f}, porque restamos el descuento del precio original.",
            f"Paso 4: Para {cantidad} unidades, el total es {precio_final:.0f} × {cantidad} = ${total:.0f} según la multiplicación.",
        ],
        "conclusion": f"El total a pagar es ${total:.0f}.",
        "verificacion": f"({precio} * (1 - {desc_pct}/100)) * {cantidad} == {total:.0f}",
    }


def _es_velocidad():
    vehiculo = random.choice(["un auto", "un tren", "un autobús", "una bicicleta", "una moto"])
    vel = random.choice([40, 50, 60, 80, 90, 100, 120])
    horas = random.choice([2, 3, 4, 5])
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


def _es_proporcion():
    item = random.choice(["litros de pintura", "kilos de harina", "metros de cable", "litros de gasolina", "kilos de cemento"])
    cant1 = random.randint(2, 5)
    resultado1 = random.randint(10, 50)
    cant2 = random.randint(6, 15)
    resultado2 = resultado1 * cant2 // cant1
    # Adjust so it's exact
    resultado2 = (resultado1 * cant2) // cant1
    if resultado1 * cant2 % cant1 != 0:
        cant2 = cant1 * random.randint(2, 4)
        resultado2 = resultado1 * cant2 // cant1
    return {
        "pregunta": f"Si {cant1} {item} alcanzan para {resultado1} m², ¿cuántos m² se cubren con {cant2} {item}?",
        "razonamiento": [
            f"Paso 1: Sabemos que {cant1} {item} cubren {resultado1} m² según el enunciado del problema.",
            f"Paso 2: Calculamos el rendimiento por unidad: {resultado1} ÷ {cant1} = {resultado1/cant1:.1f} m² por unidad, ya que dividimos total entre cantidad.",
            f"Paso 3: Para {cant2} unidades, multiplicamos {resultado1/cant1:.1f} × {cant2} = {resultado2} m², porque el rendimiento es proporcional.",
        ],
        "conclusion": f"Con {cant2} {item} se cubren {resultado2} m².",
        "verificacion": f"{resultado1} * {cant2} // {cant1} == {resultado2}",
    }


def _es_porcentaje():
    contexto = random.choice([
        ("una clase", "estudiantes", "aprobaron el examen"),
        ("una empresa", "empleados", "trabajan de forma remota"),
        ("una ciudad", "habitantes", "usan transporte público"),
        ("un grupo", "personas", "hablan más de un idioma"),
    ])
    total = random.choice([40, 50, 80, 100, 120, 200, 250])
    pct = random.choice([20, 25, 30, 40, 60, 75])
    cantidad = total * pct // 100
    return {
        "pregunta": f"En {contexto[0]} de {total} {contexto[1]}, el {pct}% {contexto[2]}. ¿Cuántos son?",
        "razonamiento": [
            f"Paso 1: El total de {contexto[1]} es {total} según el enunciado del problema.",
            f"Paso 2: El porcentaje que {contexto[2]} es {pct}%, dato que nos proporciona el problema.",
            f"Paso 3: Calculamos {total} × {pct}/100 = {cantidad}, ya que multiplicamos el total por la fracción decimal.",
        ],
        "conclusion": f"{cantidad} {contexto[1]} {contexto[2]}.",
        "verificacion": f"{total} * {pct} // 100 == {cantidad}",
    }


def gen_es_logica(seed: int) -> dict:
    random.seed(seed)
    templates = [
        _es_silogismo,
        _es_modus_ponens,
        _es_modus_tollens,
        _es_disyuncion,
        _es_transitividad,
    ]
    return random.choice(templates)()


def _es_silogismo():
    casos = [
        ("mamíferos", "vertebrados", "ballenas", "mamíferos"),
        ("aves", "animales con plumas", "pingüinos", "aves"),
        ("insectos", "artrópodos", "hormigas", "insectos"),
        ("reptiles", "animales de sangre fría", "lagartos", "reptiles"),
        ("árboles frutales", "plantas", "manzanos", "árboles frutales"),
        ("felinos", "carnívoros", "leones", "felinos"),
        ("peces", "vertebrados acuáticos", "salmones", "peces"),
    ]
    A, B, C, D = random.choice(casos)
    return {
        "pregunta": f"Si todos los {A} son {B}, y los {C} son {D}, ¿los {C} son {B}?",
        "razonamiento": [
            f"Paso 1: La primera premisa establece que todos los {A} pertenecen a la categoría de {B}.",
            f"Paso 2: La segunda premisa nos dice que los {C} son {D}, es decir, pertenecen al grupo de los {A}.",
            f"Paso 3: Aplicando silogismo categórico, los {C} son {B} porque son {A} y todos los {A} son {B}.",
        ],
        "conclusion": f"Sí, los {C} son {B} por transitividad lógica.",
    }


def _es_modus_ponens():
    casos = [
        ("llueve", "el suelo se moja", "está lloviendo", "el suelo está mojado"),
        ("hay fuego", "hay humo", "hay fuego en la cocina", "hay humo en la cocina"),
        ("es de día", "hay luz solar", "es mediodía", "hay luz solar"),
        ("la temperatura baja de 0°C", "el agua se congela", "la temperatura es -5°C", "el agua se congela"),
        ("un metal se calienta", "se dilata", "el hierro se calentó", "el hierro se dilató"),
    ]
    p, q, hp, hq = random.choice(casos)
    return {
        "pregunta": f"Si {p}, entonces {q}. Sabemos que {hp}. ¿Qué podemos concluir?",
        "razonamiento": [
            f"Paso 1: La premisa condicional establece que si {p}, entonces {q} como regla general.",
            f"Paso 2: Sabemos que {hp}, lo cual satisface la condición del antecedente según el enunciado.",
            f"Paso 3: Aplicando modus ponens, dado que el antecedente es verdadero, el consecuente también lo es.",
        ],
        "conclusion": f"Podemos concluir que {hq}.",
    }


def _es_modus_tollens():
    casos = [
        ("un animal es mamífero", "tiene sangre caliente", "no tiene sangre caliente", "no es mamífero"),
        ("una figura es un cuadrado", "tiene 4 lados iguales", "no tiene 4 lados iguales", "no es un cuadrado"),
        ("un número es par", "es divisible por 2", "no es divisible por 2", "no es par"),
    ]
    p, q, nq, np_ = random.choice(casos)
    return {
        "pregunta": f"Si {p}, entonces {q}. Sabemos que {nq}. ¿Qué podemos concluir?",
        "razonamiento": [
            f"Paso 1: La premisa condicional establece que si {p}, entonces necesariamente {q}.",
            f"Paso 2: Observamos que {nq}, lo cual niega el consecuente de la premisa condicional.",
            f"Paso 3: Aplicando modus tollens, dado que el consecuente es falso, el antecedente también lo es.",
        ],
        "conclusion": f"Podemos concluir que {np_}.",
    }


def _es_disyuncion():
    casos = [
        ("va al cine", "va al teatro", "no fue al cine", "fue al teatro"),
        ("estudia medicina", "estudia derecho", "no estudia medicina", "estudia derecho"),
        ("toma el tren", "toma el autobús", "no tomó el tren", "tomó el autobús"),
        ("come carne", "come pescado", "no come carne", "come pescado"),
    ]
    a, b, na, cb = random.choice(casos)
    return {
        "pregunta": f"María {a} o {b}. Sabemos que {na}. ¿Qué hizo María?",
        "razonamiento": [
            f"Paso 1: La premisa establece una disyunción: María {a} o {b}, según el enunciado.",
            f"Paso 2: Sabemos que María {na}, lo cual elimina la primera opción de la disyunción.",
            f"Paso 3: Dado que una de las dos opciones debe ser verdadera y la primera es falsa, entonces María {cb}.",
        ],
        "conclusion": f"María {cb}.",
    }


def _es_transitividad():
    casos = [
        ("A es mayor que B", "B es mayor que C", "A es mayor que C"),
        ("Pedro es más alto que Juan", "Juan es más alto que Luis", "Pedro es más alto que Luis"),
        ("el hierro es más denso que el aluminio", "el aluminio es más denso que la madera", "el hierro es más denso que la madera"),
        ("la empresa X factura más que Y", "Y factura más que Z", "X factura más que Z"),
    ]
    p1, p2, concl = random.choice(casos)
    return {
        "pregunta": f"Si {p1} y {p2}, ¿qué relación hay entre el primero y el último?",
        "razonamiento": [
            f"Paso 1: La primera premisa establece que {p1} como dato del problema.",
            f"Paso 2: La segunda premisa nos dice que {p2}, lo cual conecta el segundo con el tercer elemento.",
            f"Paso 3: Aplicando la propiedad transitiva de la relación de orden, podemos concluir que {concl}.",
        ],
        "conclusion": f"{concl} por transitividad.",
    }


def gen_es_causal(seed: int) -> dict:
    random.seed(seed)
    templates = [
        _es_causal_ecologia,
        _es_causal_economia,
        _es_causal_salud,
        _es_causal_fisica,
        _es_causal_social,
    ]
    return random.choice(templates)()


def _es_causal_ecologia():
    cadenas = [
        ("se talan los bosques de una región",
         "Esto provoca que el suelo pierde la protección de las raíces de los árboles contra la erosión.",
         "En consecuencia, las lluvias arrastran la capa fértil del suelo hacia los ríos y arroyos.",
         "Esto causa que los ríos se llenen de sedimentos, reduciendo la calidad del agua potable.",
         "La deforestación degrada tanto el suelo como las fuentes de agua de la región."),
        ("se introduce una especie invasora de pez en un lago",
         "Esto significa que los peces nativos enfrentan competencia directa por alimento y espacio.",
         "Dado que la especie invasora se reproduce más rápido, desplaza gradualmente a las especies nativas.",
         "En consecuencia, los depredadores que dependían de peces nativos pierden su fuente principal de alimento.",
         "La introducción de la especie invasora altera toda la cadena alimenticia del lago."),
        ("aumenta la temperatura global en 2°C",
         "Esto provoca que los glaciares polares se derriten a un ritmo acelerado.",
         "En consecuencia, el nivel del mar sube e inunda las zonas costeras bajas de todo el mundo.",
         "Esto causa que millones de personas deban migrar hacia zonas más elevadas y seguras.",
         "El calentamiento global desencadena una crisis migratoria y ambiental a escala global."),
    ]
    preg, p1, p2, p3, concl = random.choice(cadenas)
    return {
        "pregunta": f"¿Qué consecuencias tiene que {preg}?",
        "razonamiento": [
            f"Paso 1: {p1}",
            f"Paso 2: {p2}",
            f"Paso 3: {p3}",
        ],
        "conclusion": concl,
    }


def _es_causal_economia():
    cadenas = [
        ("el banco central sube las tasas de interés",
         "Esto significa que los préstamos se vuelven más caros para empresas y consumidores.",
         "En consecuencia, las personas gastan menos y las empresas reducen sus inversiones.",
         "Esto causa que la demanda agregada disminuya, lo cual frena el crecimiento económico.",
         "La subida de tasas reduce la inflación pero desacelera la economía."),
        ("un país devalúa su moneda frente al dólar",
         "Esto implica que los productos importados se vuelven más caros para los consumidores locales.",
         "Como resultado, las exportaciones del país se vuelven más competitivas en mercados internacionales.",
         "En consecuencia, la balanza comercial mejora pero el poder adquisitivo interno se reduce.",
         "La devaluación beneficia a exportadores pero perjudica a importadores y consumidores."),
        ("aumenta el precio del petróleo un 50%",
         "Esto provoca que los costos de transporte y logística se incrementan significativamente.",
         "Como resultado, los precios de todos los productos que requieren transporte también suben.",
         "Esto causa que la inflación general se acelere y el poder de compra de los hogares disminuya.",
         "El encarecimiento del petróleo genera una ola inflacionaria en toda la economía."),
    ]
    preg, p1, p2, p3, concl = random.choice(cadenas)
    return {
        "pregunta": f"¿Qué sucede cuando {preg}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }


def _es_causal_salud():
    cadenas = [
        ("una persona no duerme lo suficiente durante semanas",
         "Sabemos que la falta de sueño reduce la producción de hormonas que regulan el sistema inmune.",
         "Esto significa que el cuerpo pierde capacidad para combatir infecciones y enfermedades comunes.",
         "En consecuencia, la persona enferma con más frecuencia y su recuperación es más lenta.",
         "La privación crónica de sueño debilita el sistema inmunológico y aumenta la vulnerabilidad a enfermedades."),
        ("una persona deja de hacer ejercicio por meses",
         "Sabemos que la inactividad prolongada reduce la masa muscular y la capacidad cardiovascular.",
         "Esto provoca que el metabolismo basal disminuya y el cuerpo queme menos calorías en reposo.",
         "Como resultado, la persona tiende a acumular grasa corporal y pierde resistencia física.",
         "El sedentarismo prolongado deteriora la composición corporal y la salud cardiovascular."),
    ]
    preg, p1, p2, p3, concl = random.choice(cadenas)
    return {
        "pregunta": f"¿Qué consecuencias tiene que {preg}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }


def _es_causal_fisica():
    cadenas = [
        ("se calienta un gas en un recipiente cerrado",
         "Sabemos que al aumentar la temperatura las moléculas del gas se mueven con mayor velocidad.",
         "Esto significa que las moléculas chocan más frecuentemente contra las paredes del recipiente.",
         "En consecuencia, la presión del gas dentro del recipiente aumenta proporcionalmente a la temperatura.",
         "Calentar un gas en un recipiente cerrado aumenta su presión, según la ley de Gay-Lussac."),
        ("se aplica una fuerza constante a un objeto sobre una superficie sin fricción",
         "Sabemos que la segunda ley de Newton establece que fuerza es igual a masa por aceleración.",
         "Esto implica que el objeto experimenta una aceleración constante en la dirección de la fuerza.",
         "Como resultado, la velocidad del objeto aumenta de forma lineal con el tiempo transcurrido.",
         "El objeto acelera uniformemente mientras se mantenga la fuerza aplicada."),
    ]
    preg, p1, p2, p3, concl = random.choice(cadenas)
    return {
        "pregunta": f"¿Qué ocurre cuando {preg}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }


def _es_causal_social():
    cadenas = [
        ("se implementa educación gratuita universitaria en un país",
         "Esto significa que las barreras económicas para acceder a la universidad se eliminan.",
         "En consecuencia, más jóvenes de familias de bajos ingresos pueden obtener títulos profesionales.",
         "Esto causa que la movilidad social aumente y la desigualdad de ingresos se reduzca a largo plazo.",
         "La educación gratuita aumenta la movilidad social y reduce la desigualdad económica."),
        ("una ciudad prohíbe los autos en su centro histórico",
         "Esto implica que los residentes y visitantes deben usar transporte público o caminar.",
         "Como resultado, la contaminación del aire en el centro disminuye significativamente.",
         "En consecuencia, la calidad de vida de los residentes mejora y el turismo peatonal aumenta.",
         "La peatonalización reduce la contaminación y mejora la experiencia urbana."),
    ]
    preg, p1, p2, p3, concl = random.choice(cadenas)
    return {
        "pregunta": f"¿Qué consecuencias tiene que {preg}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }


def gen_es_analogia(seed: int) -> dict:
    random.seed(seed)
    analogias = [
        ("el sistema inmunológico y un ejército",
         "La piel funciona como las murallas de una fortaleza, ya que ambas bloquean la entrada de invasores.",
         "Los glóbulos blancos actúan como soldados de patrulla, porque detectan y atacan agentes extraños.",
         "Los anticuerpos son como órdenes de captura específicas, dado que cada uno reconoce un enemigo particular.",
         "Ambos sistemas comparten una estructura de barrera, fuerza activa y respuesta específica."),
        ("el cerebro humano y una computadora",
         "Las neuronas funcionan como transistores, ya que ambas procesan señales eléctricas de entrada y salida.",
         "La memoria a corto plazo es análoga a la memoria RAM, porque ambas almacenan información temporal de trabajo.",
         "El aprendizaje equivale a actualizar el software, dado que ambos procesos modifican las conexiones o instrucciones existentes.",
         "Ambos sistemas procesan información, almacenan datos y se adaptan, aunque con mecanismos diferentes."),
        ("una empresa y un organismo vivo",
         "El departamento financiero funciona como el sistema circulatorio, ya que ambos distribuyen recursos vitales.",
         "Los empleados son como las células, porque cada uno cumple una función especializada dentro del conjunto.",
         "La dirección actúa como el sistema nervioso central, dado que coordina y toma decisiones para todo el organismo.",
         "Ambos sistemas tienen componentes especializados, distribución de recursos y coordinación central."),
        ("el ciclo del agua y el ciclo económico del dinero",
         "La evaporación es como el ahorro, ya que en ambos casos los recursos se retiran temporalmente de la circulación.",
         "La precipitación equivale al gasto público, porque en ambos los recursos vuelven a circular en el sistema.",
         "Los ríos funcionan como los bancos, dado que ambos canalizan y distribuyen los recursos hacia donde se necesitan.",
         "Ambos ciclos muestran acumulación, liberación y distribución continua de un recurso limitado."),
        ("una biblioteca y un motor de búsqueda web",
         "El catálogo de la biblioteca funciona como el índice del buscador, ya que ambos organizan referencias al contenido.",
         "Los bibliotecarios actúan como el algoritmo de ranking, porque ambos evalúan la relevancia para el usuario.",
         "Las estanterías son como los servidores, dado que ambos almacenan físicamente la información disponible.",
         "Ambos sistemas organizan, clasifican y facilitan el acceso a grandes volúmenes de información."),
        ("el ADN y un programa informático",
         "Los genes funcionan como funciones del código, ya que cada uno contiene instrucciones para una tarea específica.",
         "Las mutaciones son análogas a los bugs, porque ambas son cambios no planificados que alteran el resultado esperado.",
         "La transcripción del ADN equivale a la compilación, dado que ambos procesos convierten instrucciones en productos ejecutables.",
         "Ambos sistemas codifican instrucciones, pueden tener errores y requieren un proceso de traducción para funcionar."),
    ]
    preg_tema, p1, p2, p3, concl = random.choice(analogias)
    return {
        "pregunta": f"¿En qué se parecen {preg_tema}?",
        "razonamiento": [f"Paso 1: {p1}", f"Paso 2: {p2}", f"Paso 3: {p3}"],
        "conclusion": concl,
    }


# =====================================================================
# ENGLISH TEMPLATES
# =====================================================================

def gen_en_math(seed: int) -> dict:
    random.seed(seed)
    templates = [_en_shopping, _en_distance, _en_discount, _en_percentage, _en_ratio]
    return random.choice(templates)()


def _en_shopping():
    item = random.choice(["notebooks", "pens", "shirts", "water bottles", "bags of rice", "movie tickets", "pairs of socks", "boxes of cereal"])
    price = random.randint(3, 45)
    qty = random.randint(2, 12)
    total = price * qty
    return {
        "pregunta": f"You buy {qty} {item} at ${price} each. How much do you spend in total?",
        "razonamiento": [
            f"Step 1: The problem states that each unit of {item} costs ${price} at the store.",
            f"Step 2: We need to calculate the total cost for {qty} units according to the problem.",
            f"Step 3: We calculate {qty} × {price} = {total}, since total cost equals quantity times unit price.",
        ],
        "conclusion": f"The total cost is ${total}.",
        "verificacion": f"{qty} * {price} == {total}",
    }


def _en_distance():
    vehicle = random.choice(["a car", "a train", "a bus", "a cyclist", "a motorcycle"])
    speed = random.choice([40, 50, 60, 80, 100, 120])
    hours = random.choice([2, 3, 4, 5])
    dist = speed * hours
    return {
        "pregunta": f"If {vehicle} travels at {speed} km/h for {hours} hours, what distance does it cover?",
        "razonamiento": [
            f"Step 1: We know that {vehicle} maintains a constant speed of {speed} km/h according to the problem.",
            f"Step 2: The travel time is {hours} hours, which is the duration given in the problem statement.",
            f"Step 3: We calculate distance as speed × time = {speed} × {hours} = {dist} km, since distance equals rate times time.",
        ],
        "conclusion": f"The distance covered is {dist} km.",
        "verificacion": f"{speed} * {hours} == {dist}",
    }


def _en_discount():
    item = random.choice(["a jacket", "a laptop bag", "a pair of shoes", "a watch", "a backpack"])
    price = random.choice([40, 50, 60, 80, 100, 120, 150])
    disc = random.choice([10, 15, 20, 25, 30])
    qty = random.randint(1, 4)
    savings = price * disc // 100
    final = price - savings
    total = final * qty
    return {
        "pregunta": f"A store offers {disc}% off on {item} priced at ${price}. If you buy {qty}, how much do you pay?",
        "razonamiento": [
            f"Step 1: The original price of {item} is ${price} according to the problem statement.",
            f"Step 2: The {disc}% discount means we save {price} × {disc}/100 = ${savings} per unit, since we multiply by the discount rate.",
            f"Step 3: The discounted price is {price} - {savings} = ${final}, because we subtract the savings from the original price.",
            f"Step 4: For {qty} units the total is {final} × {qty} = ${total}, since we multiply unit price by quantity.",
        ],
        "conclusion": f"The total cost is ${total}.",
        "verificacion": f"({price} - {price} * {disc} // 100) * {qty} == {total}",
    }


def _en_percentage():
    contexts = [
        ("a school", "students", "passed the exam"),
        ("a company", "employees", "work remotely"),
        ("a survey", "respondents", "prefer online shopping"),
        ("a city", "residents", "use public transit"),
    ]
    ctx = random.choice(contexts)
    total = random.choice([50, 80, 100, 120, 200, 250, 400])
    pct = random.choice([20, 25, 30, 40, 50, 60, 75])
    result = total * pct // 100
    return {
        "pregunta": f"In {ctx[0]} with {total} {ctx[1]}, {pct}% {ctx[2]}. How many is that?",
        "razonamiento": [
            f"Step 1: The total number of {ctx[1]} is {total} according to the problem statement.",
            f"Step 2: We need to find {pct}% of {total}, which means we multiply by the decimal equivalent.",
            f"Step 3: We calculate {total} × {pct}/100 = {result}, since percentage means parts per hundred.",
        ],
        "conclusion": f"{result} {ctx[1]} {ctx[2]}.",
        "verificacion": f"{total} * {pct} // 100 == {result}",
    }


def _en_ratio():
    items = random.choice(["liters of paint", "bags of fertilizer", "rolls of wallpaper", "boxes of tiles"])
    amt1 = random.randint(2, 5)
    cov1 = random.randint(10, 40)
    mult = random.randint(2, 4)
    amt2 = amt1 * mult
    cov2 = cov1 * mult
    return {
        "pregunta": f"If {amt1} {items} cover {cov1} m², how many m² can {amt2} {items} cover?",
        "razonamiento": [
            f"Step 1: We know that {amt1} {items} cover {cov1} m² according to the problem statement.",
            f"Step 2: We calculate the coverage per unit: {cov1} ÷ {amt1} = {cov1/amt1:.1f} m² per unit, since we divide total by quantity.",
            f"Step 3: For {amt2} units we multiply {cov1/amt1:.1f} × {amt2} = {cov2} m², because coverage scales proportionally.",
        ],
        "conclusion": f"{amt2} {items} cover {cov2} m².",
        "verificacion": f"{cov1} * {amt2} // {amt1} == {cov2}",
    }


def gen_en_logic(seed: int) -> dict:
    random.seed(seed)
    templates = [_en_syllogism, _en_modus_ponens, _en_modus_tollens, _en_disjunction, _en_transitivity]
    return random.choice(templates)()


def _en_syllogism():
    cases = [
        ("mammals", "vertebrates", "dolphins", "mammals"),
        ("birds", "animals with feathers", "eagles", "birds"),
        ("insects", "arthropods", "butterflies", "insects"),
        ("reptiles", "cold-blooded animals", "crocodiles", "reptiles"),
        ("fungi", "organisms without chlorophyll", "mushrooms", "fungi"),
    ]
    A, B, C, D = random.choice(cases)
    return {
        "pregunta": f"All {A} are {B}. {C} are {D}. Are {C} also {B}?",
        "razonamiento": [
            f"Step 1: The first premise establishes that all {A} belong to the category of {B}.",
            f"Step 2: The second premise tells us that {C} are {D}, which means they are a type of {A}.",
            f"Step 3: Applying categorical syllogism, {C} are {B} because they are {A} and all {A} are {B}.",
        ],
        "conclusion": f"Yes, {C} are {B} by logical transitivity.",
    }


def _en_modus_ponens():
    cases = [
        ("it rains", "the streets get wet", "it is raining", "the streets are wet"),
        ("a metal is heated", "it expands", "the iron bar was heated", "the iron bar expanded"),
        ("the sun sets", "it gets dark", "the sun has set", "it is dark outside"),
        ("you study hard", "you pass the exam", "Maria studied hard", "Maria passed the exam"),
    ]
    p, q, hp, hq = random.choice(cases)
    return {
        "pregunta": f"If {p}, then {q}. We know that {hp}. What can we conclude?",
        "razonamiento": [
            f"Step 1: The conditional premise establishes that if {p}, then {q} as a general rule.",
            f"Step 2: We know that {hp}, which satisfies the antecedent according to the given information.",
            f"Step 3: Applying modus ponens, since the antecedent is true, the consequent must also be true.",
        ],
        "conclusion": f"We can conclude that {hq}.",
    }


def _en_modus_tollens():
    cases = [
        ("an animal is a fish", "it lives in water", "it does not live in water", "it is not a fish"),
        ("a shape is a circle", "it has no corners", "it has corners", "it is not a circle"),
        ("a number is even", "it is divisible by 2", "it is not divisible by 2", "it is not even"),
    ]
    p, q, nq, np_ = random.choice(cases)
    return {
        "pregunta": f"If {p}, then {q}. We observe that {nq}. What follows?",
        "razonamiento": [
            f"Step 1: The conditional premise states that if {p}, then necessarily {q}.",
            f"Step 2: We observe that {nq}, which negates the consequent of the conditional statement.",
            f"Step 3: Applying modus tollens, since the consequent is false, the antecedent must also be false.",
        ],
        "conclusion": f"We conclude that {np_}.",
    }


def _en_disjunction():
    cases = [
        ("takes the bus", "takes the train", "did not take the bus", "took the train"),
        ("studies physics", "studies chemistry", "does not study physics", "studies chemistry"),
        ("goes to the park", "goes to the museum", "did not go to the park", "went to the museum"),
    ]
    a, b, na, cb = random.choice(cases)
    return {
        "pregunta": f"Tom {a} or {b}. We know he {na}. What did Tom do?",
        "razonamiento": [
            f"Step 1: The premise establishes a disjunction: Tom {a} or {b} according to the problem.",
            f"Step 2: We know that Tom {na}, which eliminates the first option from the disjunction.",
            f"Step 3: Since one of the two options must be true and the first is false, Tom {cb} by elimination.",
        ],
        "conclusion": f"Tom {cb}.",
    }


def _en_transitivity():
    cases = [
        ("A is taller than B", "B is taller than C", "A is taller than C"),
        ("iron is denser than aluminum", "aluminum is denser than wood", "iron is denser than wood"),
        ("Tokyo is larger than London", "London is larger than Paris", "Tokyo is larger than Paris"),
    ]
    p1, p2, concl = random.choice(cases)
    return {
        "pregunta": f"If {p1}, and {p2}, what is the relationship between the first and last?",
        "razonamiento": [
            f"Step 1: The first premise establishes that {p1} as a given fact in the problem.",
            f"Step 2: The second premise tells us that {p2}, which connects the second to the third element.",
            f"Step 3: Applying the transitive property of ordering, we can conclude that {concl}.",
        ],
        "conclusion": f"{concl}, by transitivity.",
    }


def gen_en_causal(seed: int) -> dict:
    random.seed(seed)
    chains = [
        ("deforestation increases in a tropical region",
         "This causes the soil to lose the protective root systems that previously held it in place.",
         "As a result, heavy rains wash away the fertile topsoil into rivers and streams downstream.",
         "Consequently, the rivers fill with sediment and the water quality deteriorates significantly.",
         "Deforestation leads to soil erosion and degradation of water sources."),
        ("a city bans single-use plastics",
         "This means that businesses must switch to biodegradable or reusable alternatives for packaging.",
         "As a result, the volume of plastic waste entering landfills and oceans decreases significantly.",
         "Consequently, marine ecosystems begin to recover as less plastic pollution enters the food chain.",
         "The plastic ban reduces waste and helps restore marine ecosystems."),
        ("interest rates rise sharply in an economy",
         "This causes borrowing to become more expensive for both consumers and businesses alike.",
         "As a result, consumer spending decreases and businesses delay investment projects and hiring.",
         "Consequently, economic growth slows down and unemployment may increase in the short term.",
         "Rising interest rates slow economic growth by reducing spending and investment."),
        ("a vaccine is widely distributed in a population",
         "This means that a large percentage of the population develops immunity to the disease.",
         "As a result, the transmission rate drops because fewer susceptible hosts are available for the virus.",
         "Consequently, the disease incidence falls dramatically and herd immunity is achieved.",
         "Widespread vaccination leads to herd immunity and controls the spread of the disease."),
        ("a major earthquake strikes near a coastal city",
         "This causes massive displacement of the ocean floor, which generates a series of powerful waves.",
         "As a result, a tsunami forms and travels at high speed toward the coastline of the city.",
         "Consequently, coastal areas experience severe flooding and infrastructure damage upon wave arrival.",
         "Earthquakes near coastlines can trigger tsunamis that cause devastating coastal flooding."),
    ]
    topic, p1, p2, p3, concl = random.choice(chains)
    return {
        "pregunta": f"What happens when {topic}?",
        "razonamiento": [f"Step 1: {p1}", f"Step 2: {p2}", f"Step 3: {p3}"],
        "conclusion": concl,
    }


def gen_en_analogy(seed: int) -> dict:
    random.seed(seed)
    analogies = [
        ("the human heart and a water pump",
         "The heart chambers function like pump compartments, since both create pressure to move fluid.",
         "The valves in the heart work like check valves in a pump, because both prevent backflow of fluid.",
         "The arteries are like distribution pipes, given that both carry pressurized fluid to where it is needed.",
         "Both systems use chambers, valves, and distribution channels to circulate fluid through a network."),
        ("a cell and a factory",
         "The nucleus acts like the management office, since both contain the instructions that guide all operations.",
         "The mitochondria function like power generators, because both convert raw materials into usable energy.",
         "The cell membrane works like the security perimeter, given that both control what enters and exits the facility.",
         "Both systems have central control, energy production, and controlled boundaries."),
        ("evolution and machine learning",
         "Genetic mutations are like random parameter changes, since both introduce variation into the system.",
         "Natural selection works like the loss function, because both determine which variations survive based on fitness.",
         "Successive generations are like training epochs, given that both iteratively improve performance over many cycles.",
         "Both processes use random variation, selective pressure, and iteration to optimize toward better solutions."),
        ("a democracy and a marketplace",
         "Voting functions like purchasing, since both allow individuals to express preferences through their choices.",
         "Political parties act like competing brands, because both try to attract the largest share of support.",
         "Elections work like market cycles, given that both periodically reset and redistribute power based on aggregate preferences.",
         "Both systems aggregate individual preferences to determine collective outcomes through competition."),
        ("the internet and the postal system",
         "Data packets are like letters, since both carry information from a sender to a specific recipient.",
         "Routers function like sorting offices, because both direct items along the most efficient path to their destination.",
         "IP addresses work like postal addresses, given that both uniquely identify where information should be delivered.",
         "Both systems route addressed packages of information through a network of intermediate nodes."),
    ]
    topic, p1, p2, p3, concl = random.choice(analogies)
    return {
        "pregunta": f"How are {topic} similar?",
        "razonamiento": [f"Step 1: {p1}", f"Step 2: {p2}", f"Step 3: {p3}"],
        "conclusion": concl,
    }


# =====================================================================
# MAIN — Generate 500 examples
# =====================================================================

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    generators_es = {
        "matematicas": gen_es_matematicas,
        "logica": gen_es_logica,
        "causal": gen_es_causal,
        "analogias": gen_es_analogia,
    }
    generators_en = {
        "math": gen_en_math,
        "logic": gen_en_logic,
        "causal": gen_en_causal,
        "analogies": gen_en_analogy,
    }

    per_cat_es = 63   # 63 × 4 = 252 ~ 250
    per_cat_en = 63   # 63 × 4 = 252 ~ 250

    accepted = 0
    rejected = 0
    rejected_reasons = {}
    records = []

    # Spanish
    for cat, gen_fn in generators_es.items():
        for i in range(per_cat_es):
            ex = gen_fn(seed=i * 100 + hash(cat) % 1000)
            ok, reason = es_valido(ex)
            if ok:
                text = format_example(ex, "es")
                record = {"text": text, "lang": "es", "category": cat}
                if "verificacion" in ex:
                    record["verificacion"] = ex["verificacion"]
                records.append(record)
                accepted += 1
            else:
                rejected += 1
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    # English
    for cat, gen_fn in generators_en.items():
        for i in range(per_cat_en):
            ex = gen_fn(seed=i * 100 + hash(cat) % 1000)
            ok, reason = es_valido(ex)
            if ok:
                text = format_example(ex, "en")
                record = {"text": text, "lang": "en", "category": cat}
                if "verificacion" in ex:
                    record["verificacion"] = ex["verificacion"]
                records.append(record)
                accepted += 1
            else:
                rejected += 1
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    # Shuffle and write
    random.shuffle(records)
    with open(OUTPUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    total = accepted + rejected
    print(f"Generated: {total}  Accepted: {accepted}  Rejected: {rejected}")
    if rejected_reasons:
        print(f"Rejection reasons: {rejected_reasons}")

    es_count = sum(1 for r in records if r["lang"] == "es")
    en_count = sum(1 for r in records if r["lang"] == "en")
    print(f"Distribution: {es_count} ES / {en_count} EN")
    print(f"Saved to: {OUTPUT}")

    # Print 5 random for review
    print(f"\n{'='*60}")
    print("5 RANDOM EXAMPLES FOR REVIEW")
    print(f"{'='*60}")
    sample = random.sample(records, min(5, len(records)))
    for i, ex in enumerate(sample, 1):
        print(f"\n--- Example {i}  [{ex['lang']}/{ex['category']}] ---")
        print(ex["text"])
        if "verificacion" in ex:
            print(f"  [verificacion: {ex['verificacion']}]")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
