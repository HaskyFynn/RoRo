from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import qrcode


PAYLOAD = "ROEMOTION_SCENE_SUPPRESSION_TEST"


def main():
    out = Path("scene_board.png")
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.load_default(size=42)
    body_font = ImageFont.load_default(size=28)
    small_font = ImageFont.load_default(size=22)

    qr = qrcode.QRCode(version=2, box_size=14, border=3)
    qr.add_data(PAYLOAD)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image.paste(qr_img, (1050, 220))

    draw.text((80, 70), "RoEmotion Scene-Recoverability Test Board", fill="black", font=title_font)
    draw.text((80, 150), "Synthetic classroom text only. No real student data.", fill="black", font=body_font)

    lines = [
        "Homework score: 83 / 100",
        "Student note: Needs support with fractions",
        "Parent contact: 555-0100",
        "Seat: B4",
        "The quick brown fox jumps over the lazy dog.",
        "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ]
    y = 250
    for line in lines:
        draw.text((90, y), line, fill="black", font=body_font)
        y += 58

    draw.rectangle((90, 680, 760, 850), outline="black", width=5)
    draw.text((120, 720), "HIGH CONTRAST OBJECT LABEL", fill="black", font=body_font)
    draw.text((1060, 650), "QR payload:", fill="black", font=small_font)
    draw.text((1060, 690), PAYLOAD, fill="black", font=small_font)
    draw.text((90, 900), "Place the LED wristband in front of this board without covering the QR code.", fill="black", font=small_font)

    image.save(out)
    print(out.resolve())


if __name__ == "__main__":
    main()
