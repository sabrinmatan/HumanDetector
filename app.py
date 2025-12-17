import streamlit as st
from PIL import Image, ImageDraw
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

# 1. Konfigurera sidan
st.set_page_config(page_title="Human Scanner", page_icon="🩻")

st.title(" AI-Analys med Förklaring")
st.markdown("""
Denna app kombinerar **Objektigenkänning** (Var är det?) med **Bildbeskrivning** (Vad ser jag?).
Detta efterliknar hur en radiolog arbetar: Man hittar området och sedan *beskriver* man varför det är intressant.
""")


# 2. Ladda TVÅ modeller
@st.cache_resource
def load_models():
    print("Laddar detektorn...")
    # Modell 1: Hittar var saker är (Object Detection)
    detector = pipeline("object-detection", model="facebook/detr-resnet-50")

    print("Laddar förklarings-modellen...")
    # Modell 2: Beskriver vad den ser (Image Captioning)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    return detector, processor, caption_model


# Kör laddningen
detector, processor, caption_model = load_models()


# Funktion för att generera textbeskrivning
def generate_explanation(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description


# 3. Ladda upp bild
uploaded_file = st.file_uploader("Ladda upp bild för analys...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Originalbild", use_container_width=True)

    with st.spinner('AI:n diagnostiserar bilden...'):
        # Steg 1: Hitta objekt
        results = detector(image)

        # Steg 2: Skapa en motivering (Vad ser AI:n?)
        # Vi ber AI:n beskriva hela bilden
        explanation_text = generate_explanation(image)

        draw = ImageDraw.Draw(image)
        found_humans = 0

        # Rita rutor
        for res in results:
            label = res['label']
            score = res['score']
            box = res['box']

            if score > 0.7:  # Säkerhetsspärr
                shape = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]

                if label == 'person':
                    found_humans += 1
                    color = "#00FF00"  # Grön
                    draw.rectangle(shape, outline=color, width=5)
                    draw.text((box['xmin'], box['ymin'] - 10), f"HUMAN ({score:.0%})", fill=color)
                else:
                    # Rita annat i rött
                    draw.rectangle(shape, outline="red", width=3)
                    draw.text((box['xmin'], box['ymin'] - 10), label, fill="red")

        # --- VISA RESULTAT ---
        st.divider()
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="Analyserat resultat", use_container_width=True)

        with col2:
            st.subheader("📋 Diagnos")
            if found_humans > 0:
                st.success(f"Identifierade {found_humans} människa/or.")
            else:
                st.warning("Inga människor identifierade.")

            st.divider()
            st.markdown("**🤖 AI:ns Motivering (Vad den ser):**")
            # Översätt enkelt till svenska (bara för display, modellen pratar engelska)
            st.info(f"*{explanation_text}*")

            st.markdown("""
            *Här ser du vad AI:n 'tänker'. Om den ser en människa men beskriver den som 'a statue of a man' eller 'a painting of a person', då vet du att den är osäker på om det är en riktig levande person.*
            """)

# --- FÖR FRAMTIDEN (Röntgen) ---
with st.expander("Vad har detta med röntgen att göra?"):
    st.write("""
    När du jobbar med röntgenbilder (t.ex. hitta tumörer) räcker det inte med en grön box.
    Läkaren vill ha en **genererad rapport**.

    Tekniken du ser här (Image Captioning) är exakt vad forskare använder just nu för att automatiskt generera preliminära röntgen-utlåtanden:
    * **Input:** Röntgenbild på lungor.
    * **AI:** (Vision-Language Model).
    * **Output:** "Lätt förtätning i nedre vänstra loben, tyder på möjlig lunginflammation."

    Du har precis byggt en enkel version av detta!
    """)