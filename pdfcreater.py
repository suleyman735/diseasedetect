from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer

from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def add_logo(canvas, doc):
    logo_path = "scap.jpg"  # Update this to your logo's path
    width, height = letter
    
    # Draw the logo
    logo_width = 1 * inch  # Adjust this value as needed
    logo_height = 1 * inch  # Adjust this value as needed
    canvas.drawImage(logo_path, width - logo_width - 10, height - logo_height - 10, logo_width, logo_height)

def create_pdf_with_logo_and_content(filename, title, report_text, image_path, data):
    # Create a SimpleDocTemplate object
    doc = SimpleDocTemplate(filename, pagesize=letter)
    
    # Create a list of elements to add to the document
    elements = []

    # Title (Add title as a centered paragraph)
    styles = getSampleStyleSheet()
    centered_title = ParagraphStyle(name='CenteredTitle', parent=styles['Title'], alignment=1)  # 1 for centered
    title_paragraph = Paragraph(title, centered_title)
    elements.append(title_paragraph)
    elements.append(Spacer(1, 0.5 * inch))
    
    # Report Text (Add the specified text)
    report_paragraph = Paragraph(report_text, styles['BodyText'])
    elements.append(report_paragraph)
    elements.append(Spacer(1, 0.5 * inch))

    # Add Image
    report_image = Image(image_path)
    report_image.drawHeight = 3 * inch  # Adjust the height as needed
    report_image.drawWidth = 6 * inch   # Adjust the width as needed
    elements.append(report_image)
    elements.append(Spacer(1, 0.5 * inch))

    # Create and style the table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    
    # Build the PDF with the logo on every page
    doc.build(elements, onFirstPage=add_logo, onLaterPages=add_logo)

# Example data
title = "Analytic Report"
report_text = (
    "Analytic Report of the “IRRIGOPTIMAL” project for the period March-July 2024, "
    "in support of Maltese agriculture and the Ministry of Agriculture, Fisheries, and "
    "Animal Rights, with the aim of reducing water usage and tackling climate change challenges."
)
image_path = "road.jpg"  # Update this to the path of your image
data = [
    ['Header 1', 'Header 2', 'Header 3'],
    ['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'],
    ['Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3'],
]

create_pdf_with_logo_and_content("dynamic_report_with_logo_and_content.pdf", title, report_text, image_path, data)



# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors
# from reportlab.pdfgen import canvas
# from reportlab.lib.units import inch

# def add_logo(canvas, doc):
#     logo_path = "scap.jpg.png"  # Update this to your logo's path
#     width, height = letter
    
#     # Draw the logo
#     logo_width = 1 * inch  # Adjust this value as needed
#     logo_height = 1 * inch  # Adjust this value as needed
#     canvas.drawImage(logo_path, width - logo_width - 10, height - logo_height - 10, logo_width, logo_height)


# def create_pdf_with_table(filename, title, data):
#     # Create a SimpleDocTemplate object
#     doc = SimpleDocTemplate(filename, pagesize=letter)
    
#     # Create a list of elements to add to the document
#     elements = []

#   # Title (Add title as a paragraph)
#     styles = getSampleStyleSheet()
#     title_paragraph = Paragraph(title, styles['Title'])
#     elements.append(title_paragraph)
#     # Title (Add title as a paragraph)
#     from reportlab.platypus import Paragraph
#     from reportlab.lib.styles import getSampleStyleSheet
    
#     styles = getSampleStyleSheet()
#     title_paragraph = Paragraph(title, styles['Title'])
#     elements.append(title_paragraph)
    
#     # Create and style the table
#     table = Table(data)
#     table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#         ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
    
#     elements.append(table)
    
#     # Build the PDF
#     doc.build(elements,onFirstPage=add_logo, onLaterPages=add_logo)

# # Example data
# title = "Dynamic PDF with Table"
# data = [
#     ['Header 1', 'Header 2', 'Header 3'],
#     ['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'],
#     ['Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3'],
# ]

# create_pdf_with_table("dynamic_table_example.pdf", title, data)






# import requests

# # Replace these with your actual credentials and URLs
# login_url = 'http://20.108.43.18/api/token/'  # The URL where you submit the login form
# data_url = 'http://20.108.43.18/api/dashboard/'

# # Your login credentials
# credentials = {
#     'username': 'Sul',
#     'password': 'West@2023'
# }

# # Create a session to persist cookies and headers
# session = requests.Session()

# # Perform the login
# response = session.post(login_url, data=credentials)

# print(response)
# # Check if login was successful
# if response.status_code == 200:
#     print("Login successful!")


#     # Now access the data page
#     data_response = session.get(data_url)
  

#     # Check if data retrieval was successful
#     if data_response.ok:
#         print("Response Status Code:", data_response.status_code)
#         print("Response Headers:", data_response.headers)
#         print("Response JSON:", data_response.json()) 
#         # Process the data
#         print("Data retrieved successfully!")
#         print(data_response)  # or use data_response.json() if the data is in JSON format
#     else:
#         print(f"Failed to retrieve data: {data_response.status_code}")
# else:
#     print(f"Login failed: {response.status_code}")