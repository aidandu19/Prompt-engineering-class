import os
import json
import tempfile
import base64
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pdfplumber
from docx import Document
from openai import OpenAI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

# Directory to store generated images
IMAGES_DIR = Path(__file__).parent / "static" / "generated"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI()  # uses OPENAI_API_KEY env var

SYSTEM_PROMPT = """You are a CV parser that extracts EVERY DETAIL from resumes/CVs. Do NOT summarize or shorten anything. Extract ALL information.
Return ONLY valid JSON with this exact structure (no markdown, no code fences):

{
  "name": "Full Name",
  "title": "Professional title / headline",
  "summary": "Write an engaging 2-3 sentence professional summary based on the CV content. This should read like a compelling personal bio.",
  "hero_dalle_prompt": "Write a DALL-E 3 prompt for a wide cinematic hero banner image that captures the ENTIRE essence of this person's career and personality. Combine visual elements from their industry, skills, interests, and achievements into one stunning composition. For example, if someone works in finance and plays rugby: 'A dramatic cinematic panorama blending a modern glass financial district skyline on the left with a pristine green rugby pitch on the right, golden hour lighting, volumetric fog, ultra wide angle, photorealistic, 4k'. Make it UNIQUE to this specific person.",
  "contact": {
    "email": "MUST extract",
    "phone": "MUST extract",
    "location": "city/country",
    "linkedin": "MUST extract the full LinkedIn URL if present",
    "github": "if found",
    "website": "if found"
  },
  "education": [
    {
      "institution": "University Name",
      "degree": "Full degree title exactly as written",
      "years": "Sep 2018 - Jun 2022",
      "logo_domain": "exeter.ac.uk",
      "description": "Include ALL details: modules, grades, thesis, relevant coursework, achievements during study"
    }
  ],
  "experience": [
    {
      "company": "Company Name",
      "role": "Exact Job Title",
      "years": "Jun 2022 - Present",
      "logo_domain": "company.com",
      "description": "DETAILED description. Include ALL bullet points and responsibilities from the CV. Do NOT summarize - list every single responsibility, achievement, and detail mentioned. Use full sentences. If the CV says 5 things about this role, include all 5.",
      "dalle_prompt": "Write a DALL-E 3 prompt capturing the ESSENCE of this specific role. Be visual and specific to the actual work done."
    }
  ],
  "certificates": [
    {
      "name": "Certificate or Award name exactly as written",
      "issuer": "Issuing organization",
      "date": "Month Year or Year",
      "description": "Any details about the certificate/award"
    }
  ],
  "awards": [
    {
      "name": "Award name exactly as written",
      "issuer": "Awarding body",
      "date": "Month Year or Year",
      "description": "Any details about what the award was for"
    }
  ],
  "skills": [
    {
      "name": "Python",
      "level": 90,
      "category": "Programming"
    }
  ],
  "languages": [
    {
      "language": "French",
      "level": "Fluent",
      "country_code": "fr",
      "flag_emoji": "🇫🇷"
    }
  ],
  "extracurriculars": [
    {
      "name": "Activity name",
      "role": "Role/position if mentioned",
      "years": "Date range if mentioned",
      "description": "FULL detailed description of involvement - include ALL details from the CV",
      "dalle_prompt": "Write a DALL-E 3 prompt for this activity"
    }
  ],
  "interests": [
    {
      "name": "Rugby",
      "dalle_prompt": "A breathtaking wide-angle photo of a pristine rugby pitch with freshly painted white lines on vivid green grass, dramatic stadium lighting, morning dew glistening, cinematic composition, ultra high quality 4k photography",
      "description": "YOU MUST write an actual compelling 1-2 sentence description about this person's passion for this interest. For example: 'A fierce competitor on the rugby pitch who thrives under pressure and values teamwork above all.' NEVER copy the example text - write something unique based on the CV."
    }
  ],
  "theme": {
    "primary_color": "pick a color hex that suits the person's industry/vibe",
    "secondary_color": "complementary accent color hex",
    "vibe": "one word: e.g. creative, corporate, technical, adventurous"
  }
}

CRITICAL RULES:
- CONTACT DETAILS ARE THE MOST IMPORTANT. You MUST extract email, phone number, and LinkedIn URL. They are ALWAYS at the top of a CV. Never skip these.
- NEVER copy example placeholder text into your output. Every description, summary, and dalle_prompt must be UNIQUE and specific to this person's actual CV content.
- For interests: you MUST write a real, personalized description for each interest. Do NOT output placeholder text like "A short evocative description for the portfolio". Write something real and specific.
- For experience descriptions: include EVERY bullet point and detail from the CV. Do NOT condense or summarize.
- For dates: ALWAYS include specific dates (month + year if available, otherwise just years). Never omit dates.
- Certificates and Awards: Extract EVERY certificate, qualification, and award mentioned anywhere in the CV.
- Extracurriculars: Any clubs, societies, volunteering, sports teams, committees - extract them ALL with full detail.
- For logo_domain, use the main website domain of the company/university (e.g. google.com, ox.ac.uk, harvard.edu)
- For dalle_prompt, write VIVID, SPECIFIC, CINEMATIC image descriptions for DALL-E 3. Think like a photographer/art director.
- For hero_dalle_prompt: this is the MOST IMPORTANT image. It should visually represent the person's entire professional identity in one stunning panoramic image. Combine elements from their career, education, and interests.
- For languages, use ISO 3166-1 alpha-2 country codes (e.g. fr, de, es, jp, cn)
- Estimate skill levels 0-100 based on how the CV presents them
- Pick theme colors that match the person's industry
- If a section has no data, use empty array []
- ADAPTABILITY: CVs vary hugely. Map whatever sections exist to the closest matching fields. NEVER skip content.
- If someone has 15+ years of experience, include ALL roles.
"""


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_cv_with_gpt(cv_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this CV and return structured JSON:\n\n{cv_text}"},
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    data = json.loads(raw)

    # Fix URLs - ensure LinkedIn, GitHub, website all have https://
    contact = data.get("contact", {})
    for key in ("linkedin", "github", "website"):
        val = contact.get(key, "")
        if val and not val.startswith("http"):
            contact[key] = "https://" + val

    return data


def generate_image(prompt: str) -> str:
    """Generate an image with DALL-E 3 and save it locally. Returns the filename."""
    # Use a hash of the prompt as filename for caching
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
    filename = f"{prompt_hash}.png"
    filepath = IMAGES_DIR / filename

    # Return cached if exists
    if filepath.exists():
        return filename

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Download the image
        import urllib.request
        urllib.request.urlretrieve(image_url, str(filepath))
        return filename
    except Exception as e:
        print(f"DALL-E generation failed for prompt: {prompt[:80]}... Error: {e}")
        return None


def generate_all_images(data: dict) -> dict:
    """Generate DALL-E images for hero, experience, extracurriculars, and interests in parallel."""
    tasks = []

    # Hero image
    if data.get("hero_dalle_prompt"):
        tasks.append(("hero", 0, data["hero_dalle_prompt"]))

    # Experience images
    for i, exp in enumerate(data.get("experience", [])):
        if exp.get("dalle_prompt"):
            tasks.append(("experience", i, exp["dalle_prompt"]))

    # Extracurricular images
    for i, extra in enumerate(data.get("extracurriculars", [])):
        if extra.get("dalle_prompt"):
            tasks.append(("extracurriculars", i, extra["dalle_prompt"]))

    # Interest images
    for i, interest in enumerate(data.get("interests", [])):
        if interest.get("dalle_prompt"):
            tasks.append(("interests", i, interest["dalle_prompt"]))

    # Generate images in parallel (max 4 at a time to avoid rate limits)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for section, idx, prompt in tasks:
            future = executor.submit(generate_image, prompt)
            futures[future] = (section, idx)

        for future in as_completed(futures):
            section, idx = futures[future]
            filename = future.result()
            if filename:
                if section == "hero":
                    data["hero_image"] = f"/static/generated/{filename}"
                else:
                    data[section][idx]["generated_image"] = f"/static/generated/{filename}"

    return data


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "cv_file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["cv_file"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            cv_text = extract_text_from_pdf(tmp_path)
        elif suffix in (".docx", ".doc"):
            cv_text = extract_text_from_docx(tmp_path)
        else:
            return "Unsupported file type. Please upload a PDF or DOCX.", 400

        if not cv_text.strip():
            return "Could not extract text from file. Please check the file.", 400

        # Step 1: Parse CV
        data = parse_cv_with_gpt(cv_text)

        # Step 2: Generate DALL-E images
        data = generate_all_images(data)

        return render_template("portfolio.html", data=data)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
