"""
main.py

Main entry point for the Spam Message Detection System.
"""

from src.utils import print_prediction


def display_menu():
    print("\n" + "=" * 55)
    print("      Spam Message Detection System")
    print("=" * 55)
    print("1. Predict a message")
    print("2. Exit")
    print("=" * 55)


def main():
    while True:
        display_menu()

        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            message = input("\nEnter your SMS message:\n\n")

            if not message.strip():
                print("\nMessage cannot be empty.")
                continue

            print_prediction(message)

        elif choice == "2":
            print("\nThank you for using the Spam Message Detection System!")
            break

        else:
            print("\nInvalid choice. Please select 1 or 2.")


if __name__ == "__main__":
    main()
