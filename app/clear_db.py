import sqlite3
import os

DB_PATH = "potholes.db"


def delete_all():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT image_path FROM potholes")
    rows = cursor.fetchall()

    for (image_path,) in rows:
        if not image_path:
            continue

        image_path = image_path.lstrip("/\\")
        full_path = os.path.join(os.getcwd(), image_path)

        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted image: {full_path}")

    cursor.execute("DELETE FROM potholes")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='potholes'")

    conn.commit()
    conn.close()

    print("Deleted all records and images")


def delete_by_image_number(number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM potholes WHERE id = ?", (number,))
    row = cursor.fetchone()

    if not row:
        print(f"Image with id {number} not found")
        return

    image_path = row[0]
    image_path = image_path.lstrip("/\ ")
    full_path = os.path.join(os.getcwd(), image_path)

    # Delete image file
    try:
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted image: {full_path}")
        else:
            print(f"Image file not found: {full_path}")
    except Exception as e:
        print(f"Error deleting image: {e}")

    # Xóa record DB
    cursor.execute(
        "DELETE FROM potholes WHERE id = ?",
        (number,)
    )
    print(f"Deleted {cursor.rowcount} row(s)")

    conn.commit()
    conn.close()


while True:
    print("\n===== MENU =====")
    print("1. Delete ALL")
    print("2. Delete by id")
    print("0. Exit")

    choice = input("Choose: ")

    if choice == "1":
        confirm = input("Delete ALL records and images? (y/n): ")
        if confirm.lower() == "y":
            delete_all()

    elif choice == "2":
        raw = input("Image id: ").strip()
        try:
            number = int(raw)
        except ValueError:
            print("Please enter a valid id")
            continue

        delete_by_image_number(number)

    elif choice == "0":
        break

    else:
        print("Invalid choice")