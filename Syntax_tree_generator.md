### Syntax Tree Generator

This project uses the nltk package for parts of speech tagging, and the generation of syntax trees. It uses customtkinter for a graphical user interface. The purpose of this project is to generate a syntax tree based on a user's given input. Additionally, the application has a game intended to enhace user's understanding of syntax by guessing syntax trees. This works by displaying a syntax tree and having the user type in the bracketed notation for that syntax tree. 


```python
import customtkinter
import nltk
import random
from nltk import pos_tag, word_tokenize, RegexpParser
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("500x500")

correct_guesses = 0

def createTree(text):
    # Find all parts of speech in above sentence
    tagged = pos_tag(word_tokenize(text))
  
    #Extract all parts of speech from any text
    chunker = RegexpParser("""
                       NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases
                       P: {<IN>}               #To extract Prepositions
                       V: {<V.*>}              #To extract Verbs
                       PP: {<p> <NP>}          #To extract Prepositional Phrases
                       VP: {<V> <NP|PP>*}      #To extract Verb Phrases
                       """)
 
    # Print all parts of speech in above sentence
    output = chunker.parse(tagged)
    print("After Extracting\n", output)
    output.draw()


def play():
    switch_to_play_screen()

def rules():
    switch_to_rules_screen()

def create_tree():
    switch_to_create_tree_screen()

def switch_to_play_screen():
    global correct_guesses
    
    clear_screen()
    play_frame = customtkinter.CTkFrame(master=root)
    play_frame.pack(fill="both", expand=True)
    
    play_label = customtkinter.CTkLabel(master=play_frame, text="Play Screen", font=("Roboto", 24))
    play_label.pack(pady=12)
    
     # Display the count of correct guesses in the top right corner
    count_label = customtkinter.CTkLabel(master=play_frame, text=f"Correct guesses: {correct_guesses}", font=("Roboto", 18), justify="right")
    count_label.pack(side="top", anchor="ne", padx=20, pady=10)
    
    back_button = customtkinter.CTkButton(master=play_frame, text="Back to Main Menu", command=switch_to_main_menu)
    back_button.pack(side="bottom", anchor="sw", padx=20, pady=20)
    
    random_number = random.randint(1, 5)
    
    # Load the image
    image_path = "/Users/zachetzkorn/Downloads/Syntax Tree Generator Project/syntaxtree" + str(random_number) + ".png"
    image = Image.open(image_path)
    
    # Resize the image to 500x200 pixels
    image = image.resize((500, 200), Image.ANTIALIAS)
    
    # Convert the image to a format that tkinter can use
    photo = ImageTk.PhotoImage(image)
    
    # Display the image in a tkinter Label
    image_label = customtkinter.CTkLabel(master=play_frame, text = "", image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    image_label.pack(pady=12)
    
    text_entry = customtkinter.CTkEntry(master=play_frame, width=150)  # Adjust width as needed
    text_entry.pack(pady=10)
    guess = text_entry.get()
    
    submit_button = customtkinter.CTkButton(master=play_frame, text="Submit", command=lambda: enterGuess(text_entry.get(), random_number))
    submit_button.pack(pady=10)

def switch_to_play_screen_after_incorrect(number):
    global correct_guesses
    
    clear_screen()
    play_frame = customtkinter.CTkFrame(master=root)
    play_frame.pack(fill="both", expand=True)
    
    play_label = customtkinter.CTkLabel(master=play_frame, text="Play Screen", font=("Roboto", 24))
    play_label.pack(pady=12)
    
      # Display the count of correct guesses in the top right corner
    count_label = customtkinter.CTkLabel(master=play_frame, text=f"Correct guesses: {correct_guesses}", font=("Roboto", 18), justify="right")
    count_label.pack(side="top", anchor="ne", padx=20, pady=10)
    
    back_button = customtkinter.CTkButton(master=play_frame, text="Back to Main Menu", command=switch_to_main_menu)
    back_button.pack(side="bottom", anchor="sw", padx=20, pady=20)
    
    random_number = random.randint(1, 5)
    
    # Load the image
    image_path = "/Users/zachetzkorn/Downloads/Syntax Tree Generator Project/syntaxtree" + str(number) + ".png"
    image = Image.open(image_path)
   
    
    # Resize the image to 500x200 pixels
    image = image.resize((500, 200), Image.ANTIALIAS)
    
    # Convert the image to a format that tkinter can use
    photo = ImageTk.PhotoImage(image)
    
    # Display the image in a tkinter Label
    image_label = customtkinter.CTkLabel(master=play_frame, text = "", image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    image_label.pack(pady=12)
    
    text_entry = customtkinter.CTkEntry(master=play_frame, width=150)  # Adjust width as needed
    text_entry.pack(pady=10)
    guess = text_entry.get()
    
    submit_button = customtkinter.CTkButton(master=play_frame, text="Submit", command=lambda: enterGuess(text_entry.get(), random_number))
    submit_button.pack(pady=10)

def enterGuess(guess, number):
    global correct_guesses

    syntaxTrees = [
        "(S (NP (DT The) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP (DT the) (NN mat)))))",
        "(S (NP (PRP I)) (VP (VBP love) (NP (DT a) (NN good) (NN book))))",
        "(S (NP (DT This) (NN house)) (VP (VBZ is) (ADJP (JJ beautiful))))",
        "(S (NP (PRP We)) (VP (VBP are) (VP (VBG learning) (PP (IN about) (NP (NNS syntax) (NNS trees))))))",
        "(S (NP (NNP John)) (VP (VBD ate) (NP (DT a) (NN sandwich))))"
    ]

    try:
        guess_tree = nltk.Tree.fromstring(guess)
        correct_tree = nltk.Tree.fromstring(syntaxTrees[number - 1])

        if guess_tree == correct_tree:
            correct_guesses += 1
            update_play_scene("Correct!", guess, number)
        else:
            update_play_scene("Incorrect, try again", guess, number)
    except Exception as e:
        update_play_scene("Invalid input, try again", guess, number)

def update_play_scene(message, guess, number):
    global correct_guesses
    # Clear the screen
    clear_screen()

    # Create a new frame for the play scene
    play_frame = customtkinter.CTkFrame(master=root)
    play_frame.pack(fill="both", expand=True)
    
    # Display the message
    message_label = customtkinter.CTkLabel(master=play_frame, text=message, font=("Roboto", 24))
    message_label.pack(pady=12)
    
    
    # Add a "Try Again" button if the guess was incorrect
    if message == "Incorrect, try again":
        try_again_button = customtkinter.CTkButton(master=play_frame, text="Try Again", command=lambda: switch_to_play_screen_after_incorrect(number))
        try_again_button.pack(side="top", anchor="s", padx=20, pady=20)
    else:
        # Add a "Next" button if the guess was correct
        next_button = customtkinter.CTkButton(master=play_frame, text="Next", command=switch_to_play_screen)
        next_button.pack(side="top", anchor="s", padx=20, pady=20)
        
def switch_to_rules_screen():
    clear_screen()
    rules_frame = customtkinter.CTkFrame(master=root)
    rules_frame.pack(fill="both", expand=True)
    
    rules_label = customtkinter.CTkLabel(master=rules_frame, text="Rules", font=("Roboto", 24))
    rules_label.pack(pady=12)
    
    rules_text = customtkinter.CTkLabel(master=rules_frame, text="To play game, make a bracketed version of the tree\nHere are the parts of speech:\n DT = Determiner \n NN = Singular Noun \n NNP = Proper Noun \n NNS = Plural Noun\n PRP = Personal Pronoun \nVBG = Gerund Verb\n VBD = Past-Tense Verb \n VBP = verb, present tense not 3rd person singular(wrap) \n VBZ = verb, present tense with 3rd person singular (bases)\n IN = Preposition \n JJ = Adjective \n The full documentation can be found here,\n https://www.guru99.com/pos-tagging-chunking-nltk.html\n But with these rules alone you can complete all syntax trees in this project\n Here is an example sentence: 'This is an example sentence'\n and here is how it would look in bracket notation:\n (S (NP (DT This) (VBZ is) (DT an) (NN example) (NN sentence)))", font=("Roboto", 12))
    rules_text.pack(pady=12)
    
    back_button = customtkinter.CTkButton(master=rules_frame, text="Back to Main Menu", command=switch_to_main_menu)
    back_button.pack(side="bottom", anchor="sw", padx=20, pady=20)

    
def switch_to_create_tree_screen():
    def create_tree_with_text():
        tree_text = text_entry.get()
        createTree(tree_text)
        
    clear_screen()
    create_tree_frame = customtkinter.CTkFrame(master=root)
    create_tree_frame.pack(fill="both", expand=True)

    create_tree_label = customtkinter.CTkLabel(master=create_tree_frame, text="Create a Tree ", font=("Roboto", 24))
    create_tree_label.pack(pady=12)
    
    text_entry = customtkinter.CTkEntry(master=create_tree_frame, width=150)  # Adjust width as needed
    text_entry.pack(pady=10)
    
    create_tree_button = customtkinter.CTkButton(master=create_tree_frame, text="Create Tree", command=create_tree_with_text)
    create_tree_button.pack(pady=10)
    
    back_button = customtkinter.CTkButton(master=create_tree_frame, text="Back to Main Menu", command=switch_to_main_menu)
    back_button.pack(side="bottom", anchor="sw", padx=20, pady=20)

def switch_to_main_menu():
    clear_screen()
    # Frame for the main screen
    main_frame = customtkinter.CTkFrame(master=root)
    main_frame.pack(fill="both", expand=True)

    label = customtkinter.CTkLabel(master=main_frame, text="Syntax Tree Guesser", font=("Roboto", 24))
    label.pack(pady=12)

    # Create a frame to contain the buttons
    button_frame = customtkinter.CTkFrame(master=main_frame)
    button_frame.pack(expand=True)

    # Create the PLAY button
    play_button = customtkinter.CTkButton(master=button_frame, text="PLAY", command=play)
    play_button.pack(side="left", padx=10)

    # Create the RULES button
    rules_button = customtkinter.CTkButton(master=button_frame, text="RULES", command=rules)
    rules_button.pack(side="left", padx=10)

    # Create the "Create Tree" button
    create_tree_button = customtkinter.CTkButton(master=button_frame, text="CREATE TREE", command=create_tree)
    create_tree_button.pack(side="left", padx=10)

    # Center the button frame horizontally and vertically
    button_frame.place(relx=0.5, rely=0.5, anchor="center")

    root.mainloop()

def clear_screen():
    # Destroy all widgets in the root window
    for widget in root.winfo_children():
        widget.destroy()

# Frame for the main screen
main_frame = customtkinter.CTkFrame(master=root)
main_frame.pack(fill="both", expand=True)

label = customtkinter.CTkLabel(master=main_frame, text="Syntax Tree Guesser", font=("Roboto", 24))
label.pack(pady=12)

# Create a frame to contain the buttons
button_frame = customtkinter.CTkFrame(master=main_frame)
button_frame.pack(expand=True)

# Create the PLAY button
play_button = customtkinter.CTkButton(master=button_frame, text="PLAY", command=play)
play_button.pack(side="left", padx=10)

# Create the RULES button
rules_button = customtkinter.CTkButton(master=button_frame, text="RULES", command=rules)
rules_button.pack(side="left", padx=10)

# Create the "Create Tree" button
create_tree_button = customtkinter.CTkButton(master=button_frame, text="CREATE TREE", command=create_tree)
create_tree_button.pack(side="left", padx=10)

# Center the button frame horizontally and vertically
button_frame.place(relx=0.5, rely=0.5, anchor="center")

root.mainloop()
```

    bgerror failed to handle background error.
        Original error: invalid command name "6078716160update"
        Error in bgerror: can't invoke "tk" command: application has been destroyed
    bgerror failed to handle background error.
        Original error: invalid command name "4417091520check_dpi_scaling"
        Error in bgerror: can't invoke "tk" command: application has been destroyed



```python
#Trees used for game portion
syntax_trees = {
    "The cat sits on the mat.": "(S (NP (DT The) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP (DT the) (NN mat)))))",
    "I love a good book.": "(S (NP (PRP I)) (VP (VBP love) (NP (DT a) (NN good) (NN book))))",
    "This house is beautiful.": "(S (NP (DT This) (NN house)) (VP (VBZ is) (ADJP (JJ beautiful))))",
    "We are learning about syntax trees.": "(S (NP (PRP We)) (VP (VBP are) (VP (VBG learning) (PP (IN about) (NP (NNS syntax) (NNS trees))))))",
    "John ate a sandwich.": "(S (NP (NNP John)) (VP (VBD ate) (NP (DT a) (NN sandwich))))"
}
```
