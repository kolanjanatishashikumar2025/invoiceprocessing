import streamlit as st
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import tempfile
#import pdf2image
import fitz # PyMuPDF

st.set_page_config(page_title="Invoice Field Extractor with Donut", layout="centered")
st.title("📄 Invoice Field Extractor (AI-powered)")

# Upload PDF
uploaded_file = st.file_uploader("Upload an invoice (PDF)", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        pdf_path = tmp_pdf.name

    # Convert first page of PDF to image
   # images = pdf2image.convert_from_path(pdf_path, dpi=200)
   # image = images[0]  # Take only first page

    # Open PDF and render page as image
doc = fitz.open(pdf_path)
page = doc.load_page(0)  # First page
pix = page.get_pixmap(dpi=200)
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    st.image(image, caption="Invoice Preview", use_column_width=True)

    with st.spinner("Extracting fields with Donut model..."):
        # Load processor and model
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Prepare image
        pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        prompt = "<s_docvqa><s_question>Extract invoice fields:<s_answer>"
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        st.markdown("### 🧾 Extracted Fields:")
        st.json(result)

        st.success("Done!")
else:
    st.info("Upload a PDF invoice to begin.")
