import sys

from pypdf import PdfReader


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf.py <pdf_file> <output_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    out_path = sys.argv[2]

    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages[:4]:
        text.append(page.extract_text() or "")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n---PAGE---\n".join(text))

if __name__ == "__main__":
    main()
