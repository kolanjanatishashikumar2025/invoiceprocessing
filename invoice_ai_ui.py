import streamlit as st
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import tempfile
import fitz  # PyMuPDF

st.set_page_config(page_title="Invoice Field Extractor with Donut", layout="centered")
st.title("📄 Invoice Field Extractor (AI-powered)")

# Upload PDF
uploaded_file = st.file_uploader("Upload an invoice (PDF)", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        pdf_path = tmp_pdf.name

    # Convert first page of PDF to image using PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # First page
    pix = page.get_pixmap(dpi=200)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    st.image(image, caption="Invoice Preview", use_container_width=True)

    with st.spinner("Extracting fields with Donut model..."):
        try:
            # Load processor and model
           # processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
           # model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            # Prepare image
            pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
            prompt = "<s><s_question>Extract invoice fields:<s_answer>"
            decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
            result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            st.markdown("### 🧾 Extracted Fields:")
            st.json(result)

            st.success("Done!")

        except Exception as e:
            st.error(f"❌ Failed to extract fields: {e}")
else:
    st.info("Upload a PDF invoice to begin.")
