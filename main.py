import tkinter as tk
from src.gui import ImageToWordApp

def main():
    root = tk.Tk()
    app = ImageToWordApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
