from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _fit_image(
    image_path: Path,
    max_width: float,
    max_height: float,
) -> ReportImage:
    """Create a proportionally resized ReportLab image."""

    image = ReportImage(str(image_path))

    width = float(image.imageWidth)
    height = float(image.imageHeight)

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    image.drawWidth = width * scale
    image.drawHeight = height * scale

    return image


def generate_pdf_report(
    image_path: Path,
    attention_path: Path,
    gradcam_path: Path,
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    severity_title: str,
    guidance: str,
) -> bytes:
    """Generate a RetinaGuard clinical-style PDF report."""

    buffer = BytesIO()

    report_id = uuid4().hex[:10].upper()
    generated_at = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="RetinaGuard AI Screening Report",
        author="RetinaGuard AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#123B66"),
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#50657A"),
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#123B66"),
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        alignment=TA_LEFT,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#243746"),
    )

    warning_style = ParagraphStyle(
        name="Warning",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#7A3E00"),
        backColor=colors.HexColor("#FFF4D6"),
        borderColor=colors.HexColor("#E3A008"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=12,
    )

    story = []

    story.append(
        Paragraph(
            "RetinaGuard AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Explainable Diabetic Retinopathy Screening Report",
            subtitle_style,
        )
    )

    metadata_table = Table(
        [
            ["Report ID", report_id],
            ["Generated on", generated_at],
            ["Model", "RETFound CFP — ViT-Large/16"],
            ["Dataset", "APTOS 2019"],
        ],
        colWidths=[4.2 * cm, 12.2 * cm],
    )

    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#123B66")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9D6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 14))

    story.append(
        Paragraph(
            "Uploaded Fundus Image",
            heading_style,
        )
    )

    story.append(
        _fit_image(
            image_path,
            max_width=15.5 * cm,
            max_height=9.5 * cm,
        )
    )

    story.append(
        Paragraph(
            "Screening Result",
            heading_style,
        )
    )

    result_table = Table(
        [
            ["Predicted DR grade", predicted_class],
            ["Model confidence", f"{confidence:.2%}"],
            ["Screening category", severity_title],
        ],
        colWidths=[5.2 * cm, 11.2 * cm],
    )

    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DCEAF7")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F8FBFD")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A8BDCC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(result_table)

    story.append(
        Paragraph(
            "Class Probabilities",
            heading_style,
        )
    )

    probability_rows = [["DR grade", "Probability"]]

    for class_name, probability in probabilities.items():
        probability_rows.append(
            [
                class_name,
                f"{probability:.2%}",
            ]
        )

    probability_table = Table(
        probability_rows,
        colWidths=[10.5 * cm, 5.9 * cm],
    )

    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B66")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9D6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F3F7FA"),
                ]),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(probability_table)

    story.append(
        Paragraph(
            "Screening Guidance",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            guidance,
            body_style,
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Explainability Analysis",
            title_style,
        )
    )

    story.append(
        Paragraph(
            (
                "The following visualizations show regions that had "
                "relatively greater influence on the model output. "
                "They are not confirmed lesion annotations."
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Attention Rollout",
            heading_style,
        )
    )

    story.append(
        _fit_image(
            attention_path,
            max_width=17 * cm,
            max_height=10.5 * cm,
        )
    )

    story.append(
        Paragraph(
            "ViT Grad-CAM",
            heading_style,
        )
    )

    story.append(
        _fit_image(
            gradcam_path,
            max_width=17 * cm,
            max_height=10.5 * cm,
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Important medical disclaimer:</b> "
                "RetinaGuard AI is a research prototype. The generated "
                "prediction and visual explanations are not a medical "
                "diagnosis, lesion segmentation, or substitute for "
                "clinical examination. Results must be reviewed by a "
                "qualified ophthalmologist."
            ),
            warning_style,
        )
    )

    document.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes