from loading import load_dataset
from visualizing import visualize, show_accuracy_loss_graphs
from training import train
from testing import start_webcam_inference
import json
import customtkinter
from PIL import Image
from threading import *
import sys

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")


class Console(customtkinter.CTkTextbox):
    def __init__(self, *args, **kwargs):
        customtkinter.CTkTextbox.__init__(self, *args, **kwargs)
        self.bind("<Destroy>", self.reset)
        self.old_stdout = sys.stdout
        sys.stdout = self

    def delete(self, *args, **kwargs):
        self.configure(state="normal")
        self.delete(*args, **kwargs)
        self.configure(state="disabled")

    def write(self, content):
        self.configure(state="normal")
        self.insert("end", content)
        self.configure(state="disabled")

    def reset(self, event):
        sys.stdout = self.old_stdout

    def flush(self):
        pass


def threading(func, args):
    t1 = Thread(target=func, args=args)
    t1.start()


def launch(gui, camgui):
    with open('data.json') as outfile:
        data = json.load(outfile)

    image_dimensions = (32, 32, 3)

    classes_csv_path = data["classes_csv"]
    sample_images_path = data["sample_images"]

    train_csv_path = data["train_csv"]
    train_images_path = data["train_images"]
    validation_csv_path = data["validation_csv"]
    validation_images_path = data["validation_images"]

    epochs = int(data["model_epochs"])
    batch_size = int(data["model_batchsize"])
    model_name = data["full_model_name"]

    condition_load = True if gui.load_switch_var.get() == "on" else False
    condition_preload = True if gui.preload_switch_var.get() == "on" else False
    condition_visualize = True if gui.visualize_switch_var.get() == "on" else False
    condition_train = True if gui.train_switch_var.get() == "on" else False
    condition_inference = True if gui.inference_switch_var.get() == "on" else False
    condition_prerecorded = True if gui.prerecorded_switch_var.get() == "on" else False

    # LOADING DATASET
    if condition_load:
        gui.main_label.configure(text="Loading Dataset")
        X_train, Y_train, X_validation, Y_validation = load_dataset(train_images_path,
                                                                    validation_images_path,
                                                                    validation_csv_path,
                                                                    image_dimensions,
                                                                    preloaded=condition_preload)

    # VISUALIZING DATASET
    if condition_visualize:
        gui.main_label.configure(text="Visualizing Dataset")
        visualize(classes_csv_path, sample_images_path, train_csv_path, validation_csv_path, image_dimensions)

    # TRAINING THE MODEL
    if condition_train:
        gui.main_label.configure(text="Training Model")
        # noinspection PyUnboundLocalVariable
        history = train(dimensions=image_dimensions,
                        X_train=X_train, Y_train=Y_train,
                        X_validation=X_validation, Y_validation=Y_validation,
                        epochs=epochs, batch_size=batch_size,
                        model_name=model_name)

        # SHOWING GRAPHS
        show_accuracy_loss_graphs(history, model_name)

    # INFERENCE
    if condition_inference:
        gui.main_label.configure(text="Inferring")
        start_webcam_inference(model_name, condition_prerecorded, camgui)


class Video_GUI(customtkinter.CTkToplevel):
    def __init__(self):
        super().__init__()

        self.title("Inference")
        self.resizable(width=False, height=False)
        self.geometry("800x600")
        self.after(250, lambda: self.iconbitmap('./assets/arrow-1.ico'))

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = customtkinter.CTkFrame(master=self, fg_color="#161b22", corner_radius=0)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.main_label = customtkinter.CTkLabel(master=self.main_frame, text="VideoStream",
                                                 font=customtkinter.CTkFont(family="Roboto Mono", size=16))
        self.main_label.grid(row=0, column=0, sticky="w", padx=20, pady=(24, 0))
        self.main_inner_frame = customtkinter.CTkFrame(master=self.main_frame, fg_color="#30363d", corner_radius=10)
        self.main_inner_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.main_inner_frame.grid_columnconfigure(0, weight=1)
        self.main_inner_frame.grid_rowconfigure(0, weight=1)
        self.image_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="", bg_color="transparent")
        self.image_label.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")


class Directories_GUI(customtkinter.CTkToplevel):
    def __init__(self):
        super().__init__()

        self.title("Edit File Directories")
        self.resizable(width=False, height=False)
        self.geometry("800x600")
        self.after(250, lambda: self.iconbitmap('./assets/arrow-1.ico'))

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = customtkinter.CTkFrame(master=self, fg_color="#161b22", corner_radius=0)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.main_label = customtkinter.CTkLabel(master=self.main_frame, text="Setup File Locations",
                                                 font=customtkinter.CTkFont(family="Roboto Mono", size=16))
        self.main_label.grid(row=0, column=0, sticky="w", padx=20, pady=(24, 0))
        self.save_button = customtkinter.CTkButton(master=self.main_frame, text="Save",
                                                   text_color="#000000",
                                                   font=customtkinter.CTkFont(family="Roboto Mono", size=16),
                                                   image=customtkinter.CTkImage(light_image=Image.open("./assets/save.png"), size=(20, 20)),
                                                   command=self.save_button_event)
        self.save_button.grid(row=0, column=1, padx=20, pady=(20, 0))
        self.main_inner_frame = customtkinter.CTkFrame(master=self.main_frame, fg_color="#30363d", corner_radius=10)
        self.main_inner_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20, columnspan=2)
        self.main_inner_frame.grid_columnconfigure(1, weight=1)

        self.classes_csv_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="Classes CSV Path",
                                                        font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.classes_csv_label.grid(row=0, column=0, padx=20, pady=(16, 0), sticky="w")
        self.classes_csv_text = customtkinter.CTkEntry(master=self.main_inner_frame, height=16,
                                                       border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                       font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.classes_csv_text.grid(row=0, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.sample_images_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="Sample Images Directory",
                                                          font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.sample_images_label.grid(row=1, column=0, padx=20, pady=(16, 0), sticky="w")
        self.sample_images_text = customtkinter.CTkEntry(master=self.main_inner_frame, height=16,
                                                         border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                         font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.sample_images_text.grid(row=1, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.line = customtkinter.CTkFrame(master=self.main_inner_frame, fg_color="#2b2d30", corner_radius=1, height=2)
        self.line.grid(row=2, column=0, padx=16, pady=(16, 0), sticky="ew", columnspan=2)
        self.train_csv_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="Train CSV Path",
                                                      font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.train_csv_label.grid(row=3, column=0, padx=20, pady=(16, 0), sticky="w")
        self.train_csv_text = customtkinter.CTkEntry(master=self.main_inner_frame, height=16,
                                                     border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                     font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.train_csv_text.grid(row=3, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.train_images_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="Training Images Directory",
                                                         font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.train_images_label.grid(row=4, column=0, padx=20, pady=(16, 0), sticky="w")
        self.train_images_text = customtkinter.CTkEntry(master=self.main_inner_frame, height=16,
                                                        border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                        font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.train_images_text.grid(row=4, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.validation_csv_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="Validation CSV Path",
                                                           font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.validation_csv_label.grid(row=5, column=0, padx=20, pady=(16, 0), sticky="w")
        self.validation_csv_text = customtkinter.CTkEntry(master=self.main_inner_frame, height=16,
                                                          border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                          font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.validation_csv_text.grid(row=5, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.validation_images_label = customtkinter.CTkLabel(master=self.main_inner_frame, text="Validation Images Directory",
                                                              font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.validation_images_label.grid(row=6, column=0, padx=20, pady=(16, 0), sticky="w")
        self.validation_images_text = customtkinter.CTkEntry(master=self.main_inner_frame, height=16,
                                                             border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                             font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.validation_images_text.grid(row=6, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.initialize_text()
        self.line = customtkinter.CTkFrame(master=self.main_inner_frame, fg_color="#2b2d30", corner_radius=1, height=2)
        self.line.grid(row=7, column=0, padx=16, pady=(16, 0), sticky="ew", columnspan=2)

    def save_button_event(self):
        with open('data.json') as outfile:
            data = json.load(outfile)
        data["classes_csv"] = self.classes_csv_text.get()
        data["sample_images"] = self.sample_images_text.get()
        data["train_csv"] = self.train_csv_text.get()
        data["train_images"] = self.train_images_text.get()
        data["validation_csv"] = self.validation_csv_text.get()
        data["validation_images"] = self.validation_images_text.get()
        with open('data.json', 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)

    def initialize_text(self):
        with open('data.json') as outfile:
            data = json.load(outfile)
        self.classes_csv_text.insert(0, data["classes_csv"])
        self.sample_images_text.insert(0, data["sample_images"])
        self.train_csv_text.insert(0, data["train_csv"])
        self.train_images_text.insert(0, data["train_images"])
        self.validation_csv_text.insert(0, data["validation_csv"])
        self.validation_images_text.insert(0, data["validation_images"])


class GUI(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Traffic Sign Classifier")
        self.resizable(width=False, height=False)
        self.geometry("1280x720")
        self.iconbitmap("./assets/arrow-1.ico")

        # self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)

        self.side_bar = customtkinter.CTkFrame(master=self, fg_color="#0d1117", corner_radius=0, width=320)
        self.side_bar.grid(row=0, column=0, sticky="nsew")
        self.side_bar.grid_columnconfigure(0, weight=1)
        self.side_bar.grid_rowconfigure(1, weight=1)
        self.side_bar.grid_propagate(False)

        self.logo_label = customtkinter.CTkLabel(master=self.side_bar, text="Traffic Sign\nClassifier",
                                                 font=customtkinter.CTkFont(family="Roboto Mono", size=22))
        self.logo_label.grid(row=0, column=0, padx=12, pady=(20, 0), sticky="nsew")
        self.launch_button = customtkinter.CTkButton(master=self.side_bar, height=32, text="Launch", text_color="#000000",
                                                     font=customtkinter.CTkFont(family="Roboto Mono", size=16),
                                                     image=customtkinter.CTkImage(light_image=Image.open("./assets/launch.png"), size=(20, 20)),
                                                     command=self.launch_button_event)
        self.launch_button.grid(row=2, column=0, padx=40, pady=8, sticky="nsew")
        self.model_label = customtkinter.CTkLabel(master=self.side_bar, text="e", text_color="#5e7373", anchor="w",
                                                  font=customtkinter.CTkFont(family="Roboto Mono", size=10))
        self.model_label.grid(row=3, column=0, padx=16, pady=0, sticky="sew")

        self.main_area = customtkinter.CTkFrame(master=self, fg_color="#161b22", corner_radius=0, width=960)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_propagate(False)

        self.main_upper_frame = customtkinter.CTkFrame(master=self.main_area, fg_color="#161b22", corner_radius=0)
        self.main_upper_frame.grid(row=0, column=0, sticky="nsew")
        self.main_upper_frame.grid_columnconfigure(1, weight=1)
        self.main_lower_frame = customtkinter.CTkFrame(master=self.main_area, fg_color="#30363d", corner_radius=10)
        self.main_lower_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self.main_lower_frame.grid_columnconfigure(1, weight=1)
        self.new_main_lower_frame = customtkinter.CTkFrame(master=self.main_area, fg_color="#30363d", corner_radius=10)
        self.new_main_lower_frame.grid_columnconfigure(0, weight=1)
        self.new_main_lower_frame.grid_rowconfigure(0, weight=1)

        self.main_label = customtkinter.CTkLabel(master=self.main_upper_frame, text="Setup Params", anchor="w",
                                                 font=customtkinter.CTkFont(family="Roboto Mono", size=18))
        self.main_label.grid(row=0, column=0, padx=20, pady=(32, 0))
        self.dir_button = customtkinter.CTkButton(master=self.main_upper_frame, text="Edit File Directories",
                                                  text_color="#000000",
                                                  font=customtkinter.CTkFont(family="Roboto Mono", size=16),
                                                  image=customtkinter.CTkImage(light_image=Image.open("./assets/browse.png"), size=(20, 20)),
                                                  command=self.edit_dir_button_event)
        self.dir_button.grid(row=0, column=2, padx=16, pady=(16, 0))
        self.dir_toplevel_window = None
        self.cam_toplevel_window = None

        self.switch_label = customtkinter.CTkLabel(master=self.main_lower_frame, text="Toggle Modes",
                                                   font=customtkinter.CTkFont(family="Roboto Mono", size=16))
        self.switch_label.grid(row=1, column=0, padx=24, pady=(24, 6), sticky="w")
        self.load_switch_var = customtkinter.StringVar(value="off")
        self.preload_switch_var = customtkinter.StringVar(value="on")
        self.visualize_switch_var = customtkinter.StringVar(value="off")
        self.train_switch_var = customtkinter.StringVar(value="off")
        self.inference_switch_var = customtkinter.StringVar(value="on")
        self.prerecorded_switch_var = customtkinter.StringVar(value="off")
        self.load_switch = customtkinter.CTkSwitch(master=self.main_lower_frame, text="Load Dataset",
                                                   font=customtkinter.CTkFont(family="Roboto Mono", size=14),
                                                   variable=self.load_switch_var, onvalue="on", offvalue="off",
                                                   command=self.load_switch_event)
        self.load_switch.grid(row=2, column=0, padx=(32, 0), pady=(12, 0), sticky="w")
        self.preload_switch = customtkinter.CTkSwitch(master=self.main_lower_frame, text="Preload",
                                                      font=customtkinter.CTkFont(family="Roboto Mono", size=14),
                                                      variable=self.preload_switch_var, onvalue="on", offvalue="off")
        self.preload_switch.grid(row=2, column=1, padx=20, pady=(12, 0), sticky="w")
        self.visualize_switch = customtkinter.CTkSwitch(master=self.main_lower_frame, text="Visualize Dataset",
                                                        font=customtkinter.CTkFont(family="Roboto Mono", size=14),
                                                        variable=self.visualize_switch_var, onvalue="on", offvalue="off",
                                                        command=self.visualize_switch_event)
        self.visualize_switch.grid(row=3, column=0, padx=32, pady=(12, 0), sticky="w")
        self.train_switch = customtkinter.CTkSwitch(master=self.main_lower_frame, text="Train Neural Network Model",
                                                    font=customtkinter.CTkFont(family="Roboto Mono", size=14),
                                                    variable=self.train_switch_var, onvalue="on", offvalue="off",
                                                    command=self.train_switch_event)
        self.train_switch.grid(row=4, column=0, padx=32, pady=(12, 0), sticky="w")
        self.inference_switch = customtkinter.CTkSwitch(master=self.main_lower_frame, text="Begin Inference",
                                                        font=customtkinter.CTkFont(family="Roboto Mono", size=14),
                                                        variable=self.inference_switch_var, onvalue="on", offvalue="off",
                                                        command=self.inference_switch_event)
        self.inference_switch.grid(row=5, column=0, padx=32, pady=(12, 0), sticky="w")
        self.prerecorded_switch = customtkinter.CTkSwitch(master=self.main_lower_frame, text="Pre-recorded",
                                                          font=customtkinter.CTkFont(family="Roboto Mono", size=14),
                                                          variable=self.prerecorded_switch_var, onvalue="on", offvalue="off")
        self.prerecorded_switch.grid(row=5, column=1, padx=20, pady=(12, 0), sticky="w")
        self.load_switch_event()
        self.visualize_switch_event()
        self.train_switch_event()
        self.inference_switch_event()
        self.line = customtkinter.CTkFrame(master=self.main_lower_frame, fg_color="#2b2d30", corner_radius=1, height=2)
        self.line.grid(row=6, column=0, padx=16, pady=(24, 16), sticky="ew", columnspan=2)
        self.model_info_label = customtkinter.CTkLabel(master=self.main_lower_frame, text="Model Information",
                                                       font=customtkinter.CTkFont(family="Roboto Mono", size=16))
        self.model_info_label.grid(row=7, column=0, padx=24, pady=(0, 6), sticky="w")

        self.model_frame = customtkinter.CTkFrame(master=self.main_lower_frame, fg_color="#30363d", corner_radius=0)
        self.model_frame.grid(row=8, column=0, columnspan=2, sticky="nsew")
        self.model_frame.grid_columnconfigure(1, weight=1)
        self.model_frame.grid_columnconfigure(3, weight=1)

        self.model_name_label = customtkinter.CTkLabel(master=self.model_frame, text="Model Name",
                                                       font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_name_label.grid(row=0, column=0, padx=24, pady=(16, 0), sticky="w")
        self.model_name_text = customtkinter.CTkEntry(master=self.model_frame, height=16,
                                                      border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                      font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_name_text.grid(row=0, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.model_ver_label = customtkinter.CTkLabel(master=self.model_frame, text="Version",
                                                      font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_ver_label.grid(row=0, column=2, padx=24, pady=(16, 0), sticky="w")
        self.model_ver_text = customtkinter.CTkEntry(master=self.model_frame, height=16,
                                                     border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                     font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_ver_text.grid(row=0, column=3, padx=16, pady=(16, 0), sticky="ew")
        self.model_epochs_label = customtkinter.CTkLabel(master=self.model_frame, text="Epochs",
                                                         font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_epochs_label.grid(row=1, column=0, padx=24, pady=(16, 0), sticky="w")
        self.model_epochs_text = customtkinter.CTkEntry(master=self.model_frame, height=16,
                                                        border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                        font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_epochs_text.grid(row=1, column=1, padx=16, pady=(16, 0), sticky="ew")
        self.model_batchsize_label = customtkinter.CTkLabel(master=self.model_frame, text="Batch Size",
                                                            font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_batchsize_label.grid(row=1, column=2, padx=24, pady=(16, 0), sticky="w")
        self.model_batchsize_text = customtkinter.CTkEntry(master=self.model_frame, height=16,
                                                           border_color="#3b3b3b", fg_color="#3b3b3b", border_width=6,
                                                           font=customtkinter.CTkFont(family="Roboto Mono", size=14))
        self.model_batchsize_text.grid(row=1, column=3, padx=16, pady=(16, 0), sticky="ew")
        self.initialize_text()
        self.full_model_name = self.model_label.cget("text")
        self.update_model_name()
        self.model_frame.bind("<Destroy>", lambda e: self.save_text())
        self.model_name_text.bind("<KeyRelease>", lambda e: self.update_model_name())
        self.model_ver_text.bind("<KeyRelease>", lambda e: self.update_model_name())
        self.model_epochs_text.bind("<KeyRelease>", lambda e: self.update_model_name())
        self.model_batchsize_text.bind("<KeyRelease>", lambda e: self.update_model_name())
        self.line = customtkinter.CTkFrame(master=self.model_frame, fg_color="#2b2d30", corner_radius=1, height=2)
        self.line.grid(row=2, column=0, padx=16, pady=(24, 16), sticky="ew", columnspan=4)

        self.console_var = customtkinter.StringVar()
        self.console_label = Console(master=self.new_main_lower_frame, fg_color="#30363d",
                                     font=customtkinter.CTkFont(family="Roboto Mono", size=12))
        self.console_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

    def update_model_name(self):
        model_name = self.model_name_text.get()
        model_ver = self.model_ver_text.get()
        model_epochs = self.model_epochs_text.get()
        model_batchsize = self.model_batchsize_text.get()
        self.full_model_name = f"{model_name}_v{model_ver}_e{model_epochs}_b{model_batchsize}.model"
        self.model_label.configure(text=self.full_model_name)

    def initialize_text(self):
        with open('data.json') as outfile:
            data = json.load(outfile)
        self.model_label.configure(text=data["full_model_name"])
        self.model_name_text.insert(0, data["model_name"])
        self.model_ver_text.insert(0, data["model_ver"])
        self.model_epochs_text.insert(0, data["model_epochs"])
        self.model_batchsize_text.insert(0, data["model_batchsize"])

    def save_text(self):
        with open('data.json') as outfile:
            data = json.load(outfile)
        data["full_model_name"] = self.full_model_name
        data["model_name"] = self.model_name_text.get()
        data["model_ver"] = self.model_ver_text.get()
        data["model_epochs"] = self.model_epochs_text.get()
        data["model_batchsize"] = self.model_batchsize_text.get()
        with open('data.json', 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)

    def launch_button_event(self):
        self.save_text()
        self.launch_button.configure(state="disabled", fg_color="#b54747", text_color_disabled="#000000")
        self.dir_button.configure(state="disabled", fg_color="#b54747", text_color_disabled="#000000")
        self.switch_frame()
        if self.inference_switch_var.get() == "on":
            if self.cam_toplevel_window is None or not self.cam_toplevel_window.winfo_exists():
                self.cam_toplevel_window = Video_GUI()
                self.cam_toplevel_window.grab_set()
            else:
                self.cam_toplevel_window.focus()
        threading(launch, (self, self.cam_toplevel_window))

    def switch_frame(self):
        self.main_lower_frame.destroy()
        self.new_main_lower_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)

    def edit_dir_button_event(self):
        if self.dir_toplevel_window is None or not self.dir_toplevel_window.winfo_exists():
            self.dir_toplevel_window = Directories_GUI()
            self.dir_toplevel_window.grab_set()
        else:
            self.dir_toplevel_window.focus()

    def load_switch_event(self):
        if self.load_switch_var.get() == "on":
            self.preload_switch.configure(state="normal")
        else:
            self.preload_switch.configure(state="disabled")

    def visualize_switch_event(self):
        if self.visualize_switch_var.get() == "on":
            self.load_switch_var.set(value="on")
            self.load_switch_event()
            self.load_switch.configure(state="disabled")
        else:
            if self.train_switch_var.get() == "off":
                self.load_switch.configure(state="normal")

    def train_switch_event(self):
        if self.train_switch_var.get() == "on":
            self.load_switch_var.set(value="on")
            self.load_switch_event()
            self.load_switch.configure(state="disabled")
        else:
            if self.visualize_switch_var.get() == "off":
                self.load_switch.configure(state="normal")

    def inference_switch_event(self):
        if self.inference_switch_var.get() == "on":
            self.load_switch_var.set(value="off")
            self.visualize_switch_var.set(value="off")
            self.train_switch_var.set(value="off")
            self.prerecorded_switch_var.set(value="off")
            self.load_switch_event()
            self.visualize_switch_event()
            self.train_switch_event()
            self.load_switch.configure(state="disabled")
            self.visualize_switch.configure(state="disabled")
            self.train_switch.configure(state="disabled")
            self.prerecorded_switch.configure(state="normal")
        else:
            self.load_switch.configure(state="normal")
            self.visualize_switch.configure(state="normal")
            self.train_switch.configure(state="normal")
            self.prerecorded_switch.configure(state="disabled")


if __name__ == '__main__':
    window = GUI()
    window.mainloop()
