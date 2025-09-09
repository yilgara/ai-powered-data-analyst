import os
import requests
from fpdf import FPDF
from PIL import Image
from datetime import datetime
from fpdf.enums import XPos, YPos


def download_dejavu_font():
    url = "https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip"
    zip_path = "freefont.zip"
    font_folder = "freefont-20120503"
    font_file = "FreeSans.ttf"

    if not os.path.exists(font_file):
        # Download the zip file if not exists
        if not os.path.exists(zip_path):
            r = requests.get(url)
            if r.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(r.content)
            else:
                raise Exception("Failed to download font zip")

        # Extract the zip and get font file
        import zipfile
        import shutil
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        extracted_font_path = os.path.join(font_folder, font_file)
        if os.path.exists(extracted_font_path):
            shutil.move(extracted_font_path, font_file)
        else:
            raise FileNotFoundError(f"{extracted_font_path} not found")
    return font_file





def create_pdf(plot_files, report_title, insights=None, output_file="gpt_data_report_new.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)



    # First page: Title and Date
    pdf.add_page()
    pdf.set_font("Times", '', 24)
    pdf.cell(0, 40, report_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    
    today = datetime.today()
    date_str = today.strftime("Date: %d %B %Y")

    pdf.set_font("Times", '', 14)
    pdf.cell(0, 10, date_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')


    if insights is None:
        insights = [""] * len(plot_files)


    num_pages = min(len(plot_files), len(insights))

    for i in range(num_pages):
        pdf.add_page()
        img_path = plot_files[i]
        # Calculate scaled height based on image dimensions
        desired_width = pdf.w - 30
        with Image.open(img_path) as im:
            orig_width, orig_height = im.size
        scaled_height = (orig_height / orig_width) * desired_width

        # Add some space before image
        pdf.ln(10)

        # Insert image at current y, and record Y position
        current_y = pdf.get_y()
        pdf.image(img_path, x=15, y=current_y, w=desired_width)

        # Move cursor below the image
        pdf.set_y(current_y + scaled_height + 10)

        # Key Insights - bold header + bullet points
        pdf.set_font("Times", '', 14)
        pdf.cell(0, 10, "Key Points:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Times", '', 12)
        # If insights[i] is a list, print bullets
        page_width = pdf.w - 2 * pdf.l_margin  # usable width within margins

        if isinstance(insights[i], (list, tuple)):
            for insight in insights[i]:
                x_start = pdf.get_x()
                pdf.multi_cell(page_width, 8, f"- {insight}")
                pdf.set_x(x_start)  # reset x so it doesn’t shift right
        else:
            pdf.multi_cell(page_width, 8, f"- {insights[i]}")
        pdf.ln(3)


    for page_num in range(1, num_pages + 1):
        pdf.page = page_num
        pdf.set_y(-15)
        pdf.set_font("Times", '', 10)
        pdf.cell(0, 10, f"Page {page_num} / {num_pages}", align='C')

    pdf.output(output_file)
    print(f"Report saved as: {output_file}")
    return output_file
