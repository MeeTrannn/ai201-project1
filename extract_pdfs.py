from pathlib import Path
import argparse
import pdfplumber


def extract_text_from_pdf(pdf_path: Path) -> str:
    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    texts.append(txt)
    except Exception as e:
        raise RuntimeError(f"Failed to read {pdf_path}: {e}")

    # join pages with double newline for readability
    return "\n\n".join(texts).strip()


def main(input_dir: Path, output_dir: Path, overwrite: bool):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    for pdf in pdf_files:
        out_path = output_dir / (pdf.stem + ".txt")
        if out_path.exists() and not overwrite:
            print(f"Skipping (exists): {out_path}")
            continue

        try:
            text = extract_text_from_pdf(pdf)
        except Exception as e:
            print(e)
            continue

        if not text:
            print(f"No extractable text in {pdf}; creating empty file")

        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract text from PDFs in a folder using pdfplumber")
    ap.add_argument("--input-dir", "-i", type=Path, default=Path("documents"), help="Folder containing PDF files")
    ap.add_argument("--output-dir", "-o", type=Path, default=Path("extracted_texts"), help="Folder to write .txt outputs")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = ap.parse_args()

    main(args.input_dir, args.output_dir, args.overwrite)
