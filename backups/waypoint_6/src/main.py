import customtkinter as ctk
from gui.app import EyeTrackingApp

def main():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = EyeTrackingApp()
    app.mainloop()

if __name__ == "__main__":
    main()