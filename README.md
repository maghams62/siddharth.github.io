# Siddharth Suresh — Personal Site & Resume RAG Chatbot

This repository contains a minimalist, recruiter-friendly personal website for **Siddharth Suresh**. The site is fully static (GitHub Pages ready), styled with Tailwind via CDN, and includes a floating chatbot that runs retrieval-augmented generation (RAG) *entirely in the browser* using TF-IDF vectors computed from the resume PDF.

## Update Your Content

All copy lives in a single data object near the bottom of `index.html`. Look for the comment `// EDIT ME` and update values inside `window.SITE_DATA`. The JavaScript renderer updates every section—hero, highlights, experience timeline, projects, patents, education, skills, contact, and navigation—based on that object.

Key spots to refresh:

- `profile.links` — swap placeholder URLs for your actual profiles.
- `experience`, `projects`, `patents`, `education`, `skills`, `contact` — tailor bullets, dates, and descriptions.
- `seo.url` — set to your deployed GitHub Pages URL for accurate OpenGraph metadata.

## Manage Resume & RAG Pack

1. Replace the placeholder resume at `assets/Siddharth_Suresh_Resume.pdf` with your latest PDF.
2. Ensure you have the dependencies:

   ```bash
   pip install PyPDF2
   ```

3. Regenerate the chatbot pack whenever the resume changes:

   ```bash
   python scripts/make_pack.py assets/Siddharth_Suresh_Resume.pdf assets/rag/resume_rag_pack.json
   ```

   The script extracts text, chunks paragraphs/bullets, tokenizes with a built-in stopword list, and writes TF-IDF vectors plus chunk norms to `assets/rag/resume_rag_pack.json`.

## Run Locally

No build step is required. Open `index.html` in any modern browser (double-click or drag-drop). Tailwind loads from its CDN, and the chatbot fetches `assets/rag/resume_rag_pack.json` from the local filesystem.

## Deploy to GitHub Pages

1. If publishing as a personal site, name the repository `<username>.github.io` and push to `main`. GitHub automatically serves it at `https://<username>.github.io/`.
2. For a project site, push to any repo, then enable **Settings → Pages** and choose the branch (typically `main`) with `/` as the root.
3. After deploying, update `seo.url` in `window.SITE_DATA` so OpenGraph metadata points to the live URL.
4. If you’ve connected the chatbot to a Hugging Face Space, make sure the endpoint URL in `index.html` matches your Space (`HF_ENDPOINT` constant near the bottom of the file). Update it whenever you rename or move the Space.

## Privacy

The chatbot runs completely on the client. All TF-IDF search and answer assembly happen in your browser; no questions or resume data ever leave the page.

## File Structure

```
index.html               # Single-page site with Tailwind, data renderer, and chatbot
README.md                # This guide
assets/
  avatar.jpg             # Placeholder avatar – replace with your own image
  Siddharth_Suresh_Resume.pdf  # Resume used to build the RAG pack
  rag/
    resume_rag_pack.json # Generated TF-IDF data (regenerate via scripts/make_pack.py)
scripts/
  make_pack.py           # CLI to convert PDF resume into TF-IDF JSON pack
```
