# 🧬 AI Human Detector & Explainable AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Live-success)

> **Ett Computer Vision-system som inte bara detekterar människor, utan också genererar en text-motivering (Captioning) för sina beslut. Byggt som en "Proof of Concept" för medicinsk bildanalys.**

---

## 🖼️ Demo & Resultat

*(Här kan du lägga in en skärmdump på när programmet har hittat en människa och skrivit en text)*
![Demo Screenshot](https://via.placeholder.com/800x400.png?text=Ladda+upp+en+screenshot+här)

---

## 🏥 Varför detta projekt? (Koppling till Röntgen/MedTech)

Mitt mål är att arbeta med **AI inom medicinsk bilddiagnostik (Röntgen/MR)**. Detta projekt är byggt för att demonstrera de två fundamentala tekniker som används för att automatisera läkarutlåtanden:

1.  **Object Detection (Var är det?):**
    * *I detta projekt:* Ritar en grön box runt människor.
    * *I vården:* Används för att markera frakturer, tumörer eller organ.
2.  **Image Captioning / XAI (Vad är det?):**
    * *I detta projekt:* Genererar en textbeskrivning (t.ex. "a woman sitting on a chair").
    * *I vården:* Motsvarar den automatiska rapporten (t.ex. "Förtätning i vänster lunglob, misstänkt pneumoni").

Genom att kombinera dessa två modeller (DETR + BLIP) visar jag hur man bygger ett system som är **Explainable (XAI)** – något som är ett krav för patientsäkerhet.

---

## ⚙️ Så fungerar det (Under huven)

Applikationen använder **Transfer Learning** med två state-of-the-art modeller från Hugging Face:

### 1. Detektorn (Facebook DETR)
Modellen **DE**tection **TR**ansformer (ResNet-50) skannar bilden.
* Om den hittar en person med >70% säkerhet, ritas en grön bounding box.
* Om den hittar djur eller objekt, markeras dessa rött.

### 2. Förklaringsmodellen (Salesforce BLIP)
Modellen **B**ootstrapping **L**anguage-**I**mage **P**retraining läser in bilden och genererar en mening som beskriver innehållet.
* Detta ger oss "Motiveringen". Om detektorn säger "Människa", men BLIP säger "a statue of a man", kan vi dra slutsatsen att det inte är en levande person.

---

## 🛠️ Teknisk Stack

* **Frontend:** Streamlit (Python)
* **AI/ML:** PyTorch, Transformers (Hugging Face)
* **Bildhantering:** Pillow (PIL), NumPy
* **Modeller:**
    * `facebook/detr-resnet-50` (Object Detection)
    * `Salesforce/blip-image-captioning-base` (Image Captioning)

---

## 🚀 Hur du kör projektet lokalt

1. **Starta appen:**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Framtida förbättringar (Roadmap)

För att ta detta närmare en skarp medicinsk applikation planerar jag att:
* [ ] Finjustera (Fine-tune) modellen på ett dataset med röntgenbilder (t.ex. ChestX-ray8).
* [ ] Lägga till stöd för DICOM-filer (standardformatet för röntgen).
* [ ] Implementera en "Heatmap" (Grad-CAM) för att visuellt visa exakt vilka pixlar AI:n tittar på.

---

## 👤 Kontakt

Utvecklad av **[Sabirin Matan]** – Aspiring AI Engineer inom MedTech.
# HumanDetector
