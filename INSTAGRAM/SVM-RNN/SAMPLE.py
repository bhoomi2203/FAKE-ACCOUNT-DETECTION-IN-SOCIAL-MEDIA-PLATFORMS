"""
Instagram Fake Account Detector - Modern Desktop Application
A beautiful, reactive GUI for fake account detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading
import re

class ModernFakeAccountDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Instagram Fake Account Detector")
        
        # Set fullscreen mode
        self.root.state('zoomed')  # For Windows
        # Alternatively, use: self.root.attributes('-fullscreen', True)
        
        self.root.resizable(True, True)
        
        # Models
        self.svm_model = None
        self.rnn_model = None
        self.scaler = None
        self.tokenizer = None
        self.max_length = 100
        self.models_loaded = False
        
        # Color scheme - Modern gradient theme
        self.colors = {
            'bg_primary': '#0f0f23',
            'bg_secondary': '#1a1a2e',
            'bg_card': '#16213e',
            'accent': '#e94560',
            'accent_light': '#ff6b8a',
            'success': '#00d4aa',
            'warning': '#ffa726',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'border': '#2d3748',
            'fake_color': '#ff4757',
            'real_color': '#2ed573',
            'scrollbar': "#2b2b35"  # Darker scrollbar color
        }
        
        # Configure root
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Create UI
        self.create_styles()
        self.create_ui()
        
        # Auto-load models
        self.root.after(100, self.auto_load_models)
        
    def create_styles(self):
        """Create custom ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=self.colors['bg_primary'])
        style.configure('Card.TFrame', background=self.colors['bg_card'], 
                       relief='flat', borderwidth=0)
        
        # Labels
        style.configure('Title.TLabel', background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'], 
                       font=('Segoe UI', 24, 'bold'))
        style.configure('Subtitle.TLabel', background=self.colors['bg_primary'],
                       foreground=self.colors['text_secondary'], 
                       font=('Segoe UI', 10))
        style.configure('Card.TLabel', background=self.colors['bg_card'],
                       foreground=self.colors['text_primary'], 
                       font=('Segoe UI', 10, 'bold'))
        style.configure('CardValue.TLabel', background=self.colors['bg_card'],
                       foreground=self.colors['text_secondary'], 
                       font=('Segoe UI', 9))
        
        # Entry
        style.configure('Modern.TEntry', fieldbackground=self.colors['bg_secondary'],
                       foreground=self.colors['text_primary'], 
                       borderwidth=0, relief='flat')
        
        # Checkbutton
        style.configure('Modern.TCheckbutton', background=self.colors['bg_card'],
                       foreground=self.colors['text_primary'])
        
    def create_ui(self):
        """Create the main UI"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_container)
        
        # Create scrollable content
        canvas = tk.Canvas(main_container, bg=self.colors['bg_primary'], 
                          highlightthickness=0)
        
        # Custom darker scrollbar
        scrollbar = tk.Scrollbar(main_container, orient="vertical", 
                                command=canvas.yview,
                                bg=self.colors['scrollbar'],
                                troughcolor=self.colors['bg_secondary'],
                                activebackground=self.colors['border'])
        
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Content sections
        self.create_input_section(scrollable_frame)
        self.create_action_section(scrollable_frame)
        self.create_result_section(scrollable_frame)
        
        # Pack scrollable area
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel
        canvas.bind_all("<MouseWheel>", 
                       lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
    def create_header(self, parent):
        """Create header section"""
        header = tk.Frame(parent, bg=self.colors['bg_primary'])
        header.pack(fill='x', pady=(0, 20))
        
        # Title with gradient effect
        title_frame = tk.Frame(header, bg=self.colors['bg_primary'])
        title_frame.pack()
        
        title = tk.Label(title_frame, text="📊 Instagram Account Analyzer",
                        font=('Segoe UI', 28, 'bold'),
                        fg=self.colors['accent'],
                        bg=self.colors['bg_primary'])
        title.pack()
        
        subtitle = tk.Label(title_frame, 
                           text="ML Based Fake Account Detection using SVM-RNN Hybrid Model",
                           font=('Segoe UI', 10),
                           fg=self.colors['text_secondary'],
                           bg=self.colors['bg_primary'])
        subtitle.pack()
        
        # Status indicator
        self.status_frame = tk.Frame(header, bg=self.colors['bg_primary'])
        self.status_frame.pack(pady=10)
        
        self.status_label = tk.Label(self.status_frame, text="🔄 Loading models...",
                                    font=('Segoe UI', 9),
                                    fg=self.colors['warning'],
                                    bg=self.colors['bg_primary'])
        self.status_label.pack()
        
    def create_input_section(self, parent):
        """Create input section with modern cards"""
        # Input container
        input_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        input_frame.pack(fill='x', pady=10)
        
        # Create two columns
        left_col = tk.Frame(input_frame, bg=self.colors['bg_primary'])
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        right_col = tk.Frame(input_frame, bg=self.colors['bg_primary'])
        right_col.pack(side='left', fill='both', expand=True, padx=(10, 0))
        
        # Account Details Card
        self.create_card(left_col, "👤 Account Details", [
            ("Username", "username"),
            ("Full Name", "fullname"),
            ("Bio/Description", "bio", True)
        ])
        
        # Statistics Card
        self.create_card(left_col, "📈 Account Statistics", [
            ("Number of Posts", "posts"),
            ("Number of Followers", "followers"),
            ("Number of Following", "following")
        ])
        
        # Flags Card
        self.create_flags_card(right_col)
        
        # Content Card
        self.create_content_card(right_col)
        
    def create_card(self, parent, title, fields):
        """Create a modern card with fields"""
        card = tk.Frame(parent, bg=self.colors['bg_card'], 
                       relief='flat', bd=0)
        card.pack(fill='x', pady=10, padx=5)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_card'])
        header.pack(fill='x', padx=15, pady=(15, 10))
        
        tk.Label(header, text=title,
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(anchor='w')
        
        # Fields
        for field in fields:
            self.create_field(card, field[0], field[1], 
                            multiline=len(field) > 2 and field[2])
        
    def create_field(self, parent, label_text, var_name, multiline=False):
        """Create an input field"""
        field_frame = tk.Frame(parent, bg=self.colors['bg_card'])
        field_frame.pack(fill='x', padx=15, pady=8)
        
        label = tk.Label(field_frame, text=label_text,
                        font=('Segoe UI', 9),
                        fg=self.colors['text_secondary'],
                        bg=self.colors['bg_card'])
        label.pack(anchor='w', pady=(0, 5))
        
        if multiline:
            widget = tk.Text(field_frame, height=3,
                           bg=self.colors['bg_secondary'],
                           fg=self.colors['text_primary'],
                           font=('Segoe UI', 10),
                           relief='flat', bd=0,
                           insertbackground=self.colors['text_primary'])
            widget.pack(fill='x', ipady=5, ipadx=10)
            setattr(self, f'{var_name}_entry', widget)
        else:
            widget = tk.Entry(field_frame,
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=('Segoe UI', 10),
                            relief='flat', bd=0,
                            insertbackground=self.colors['text_primary'])
            widget.pack(fill='x', ipady=8, ipadx=10)
            setattr(self, f'{var_name}_entry', widget)
        
    def create_flags_card(self, parent):
        """Create flags card with checkboxes"""
        card = tk.Frame(parent, bg=self.colors['bg_card'])
        card.pack(fill='x', pady=10, padx=5)
        
        # Header
        header = tk.Frame(card, bg=self.colors['bg_card'])
        header.pack(fill='x', padx=15, pady=(15, 10))
        
        tk.Label(header, text="🚩 Account Flags",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(anchor='w')
        
        # Checkboxes
        flags_frame = tk.Frame(card, bg=self.colors['bg_card'])
        flags_frame.pack(fill='x', padx=15, pady=10)
        
        self.profile_pic_var = tk.BooleanVar(value=True)
        self.external_url_var = tk.BooleanVar(value=False)
        self.private_var = tk.BooleanVar(value=False)
        
        self.create_checkbox(flags_frame, "Has Profile Picture", 
                            self.profile_pic_var)
        self.create_checkbox(flags_frame, "Has External URL", 
                            self.external_url_var)
        self.create_checkbox(flags_frame, "Private Account", 
                            self.private_var)
        
    def create_checkbox(self, parent, text, variable):
        """Create a modern checkbox"""
        cb = tk.Checkbutton(parent, text=text, variable=variable,
                           bg=self.colors['bg_card'],
                           fg=self.colors['text_primary'],
                           font=('Segoe UI', 10),
                           selectcolor=self.colors['bg_secondary'],
                           activebackground=self.colors['bg_card'],
                           activeforeground=self.colors['text_primary'],
                           relief='flat', bd=0)
        cb.pack(anchor='w', pady=5)
        
    def create_content_card(self, parent):
        """Create content card for captions and comments"""
        card = tk.Frame(parent, bg=self.colors['bg_card'])
        card.pack(fill='both', expand=True, pady=10, padx=5)
        
        # Header
        header = tk.Frame(card, bg=self.colors['bg_card'])
        header.pack(fill='x', padx=15, pady=(15, 10))
        
        tk.Label(header, text="📝 Content Analysis",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(anchor='w')
        
        # Captions
        self.create_text_field(card, 'Recent Captions (separate with "")', 
                              "captions")
        
        # Comments
        self.create_text_field(card, 'Recent Comments (separate with "")', 
                              "comments")
        
    def create_text_field(self, parent, label_text, var_name):
        """Create a text area field"""
        field_frame = tk.Frame(parent, bg=self.colors['bg_card'])
        field_frame.pack(fill='both', expand=True, padx=15, pady=8)
        
        label = tk.Label(field_frame, text=label_text,
                        font=('Segoe UI', 9),
                        fg=self.colors['text_secondary'],
                        bg=self.colors['bg_card'])
        label.pack(anchor='w', pady=(0, 5))
        
        widget = scrolledtext.ScrolledText(field_frame, height=4,
                                          bg=self.colors['bg_secondary'],
                                          fg=self.colors['text_primary'],
                                          font=('Segoe UI', 9),
                                          relief='flat', bd=0,
                                          insertbackground=self.colors['text_primary'])
        widget.pack(fill='both', expand=True)
        setattr(self, f'{var_name}_entry', widget)
        
    def create_action_section(self, parent):
        """Create action buttons section"""
        action_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        action_frame.pack(fill='x', pady=20)
        
        # Analyze button
        self.analyze_btn = tk.Button(action_frame, 
                                     text="🔍 ANALYZE ACCOUNT",
                                     command=self.analyze_account,
                                     bg=self.colors['accent'],
                                     fg=self.colors['text_primary'],
                                     font=('Segoe UI', 12, 'bold'),
                                     relief='flat', bd=0,
                                     padx=40, pady=15,
                                     cursor='hand2',
                                     activebackground=self.colors['accent_light'])
        self.analyze_btn.pack(side='left', expand=True, padx=5)
        
        # Clear button
        clear_btn = tk.Button(action_frame, 
                             text="🗑️ CLEAR",
                             command=self.clear_fields,
                             bg=self.colors['bg_card'],
                             fg=self.colors['text_primary'],
                             font=('Segoe UI', 12, 'bold'),
                             relief='flat', bd=0,
                             padx=40, pady=15,
                             cursor='hand2',
                             activebackground=self.colors['border'])
        clear_btn.pack(side='left', expand=True, padx=5)
        
    def create_result_section(self, parent):
        """Create result display section"""
        self.result_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        self.result_frame.pack(fill='both', expand=True, pady=20)
        
        # This will be populated after analysis
        
    def auto_load_models(self):
        """Automatically load models on startup"""
        def load():
            try:
                with open('svm_model.pkl', 'rb') as f:
                    self.svm_model = pickle.load(f)
                
                self.rnn_model = load_model('rnn_model.h5')
                
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                with open('tokenizer.pkl', 'rb') as f:
                    self.tokenizer = pickle.load(f)
                
                self.models_loaded = True
                self.root.after(0, self.update_status, True)
            except Exception as e:
                self.root.after(0, self.update_status, False, str(e))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def update_status(self, success, error_msg=""):
        """Update status indicator"""
        if success:
            self.status_label.config(text="✅ Models loaded successfully",
                                    fg=self.colors['success'])
            self.analyze_btn.config(state='normal')
        else:
            self.status_label.config(
                text=f"❌ Failed to load models: {error_msg}",
                fg=self.colors['fake_color'])
            self.analyze_btn.config(state='disabled')
            
    def get_entry_value(self, entry):
        """Get value from entry widget"""
        if isinstance(entry, tk.Text) or isinstance(entry, scrolledtext.ScrolledText):
            return entry.get('1.0', 'end-1c').strip()
        return entry.get().strip()
    
    def parse_quoted_strings(self, text):
        """Parse text to extract strings within double quotes"""
        # Find all strings within double quotes
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, text)
        return [match.strip() for match in matches if match.strip()]
        
    def analyze_account(self):
        """Analyze the account"""
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded. Please restart the application.")
            return
        
        # Get all input values
        try:
            username = self.get_entry_value(self.username_entry)
            fullname = self.get_entry_value(self.fullname_entry)
            bio = self.get_entry_value(self.bio_entry)
            posts = int(self.get_entry_value(self.posts_entry) or 0)
            followers = int(self.get_entry_value(self.followers_entry) or 0)
            following = int(self.get_entry_value(self.following_entry) or 0)
            
            captions_text = self.get_entry_value(self.captions_entry)
            comments_text = self.get_entry_value(self.comments_entry)
            
            # Parse quoted strings instead of splitting by |
            captions = self.parse_quoted_strings(captions_text)
            comments = self.parse_quoted_strings(comments_text)
            
            profile_pic = 1 if self.profile_pic_var.get() else 0
            external_url = 1 if self.external_url_var.get() else 0
            private = 1 if self.private_var.get() else 0
            
            if not username:
                messagebox.showwarning("Missing Input", "Please enter a username.")
                return
            
            # Show loading
            self.show_loading()
            
            # Run prediction in thread
            def predict():
                result = self.predict_account({
                    'username': username,
                    'fullname': fullname,
                    'bio': bio,
                    'posts': posts,
                    'followers': followers,
                    'following': following,
                    'captions': captions,
                    'comments': comments,
                    'profile_pic': profile_pic,
                    'external_url': external_url,
                    'private': private
                })
                self.root.after(0, self.display_results, result)
            
            thread = threading.Thread(target=predict, daemon=True)
            thread.start()
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numbers for statistics.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def show_loading(self):
        """Show loading animation"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        loading = tk.Label(self.result_frame, 
                          text="🔄 Analyzing account...",
                          font=('Segoe UI', 14),
                          fg=self.colors['accent'],
                          bg=self.colors['bg_primary'])
        loading.pack(pady=50)
        
    def predict_account(self, data):
        """Predict if account is fake or real"""
        # Calculate features
        username = data['username']
        fullname = data['fullname']
        bio = data['bio']
        
        nums_in_username = sum(c.isdigit() for c in username)
        username_length = len(username)
        nums_ratio_username = nums_in_username / username_length if username_length > 0 else 0
        
        fullname_words = len(fullname.split())
        nums_in_fullname = sum(c.isdigit() for c in fullname)
        fullname_length = len(fullname)
        nums_ratio_fullname = nums_in_fullname / fullname_length if fullname_length > 0 else 0
        
        name_equals_username = 1 if username.lower() == fullname.lower() else 0
        description_length = len(bio)
        
        csv_features = [
            data['profile_pic'],
            nums_ratio_username,
            fullname_words,
            nums_ratio_fullname,
            name_equals_username,
            description_length,
            data['external_url'],
            data['private'],
            data['posts'],
            data['followers'],
            data['following']
        ]
        
        csv_array = np.array(csv_features).reshape(1, -1)
        csv_scaled = self.scaler.transform(csv_array)
        
        # Prepare text
        text = f"{username} {fullname} {bio} "
        text += ' '.join(data['captions'])
        text += ' '.join(data['comments'])
        
        sequences = self.tokenizer.texts_to_sequences([text])
        nlp_padded = pad_sequences(sequences, maxlen=self.max_length, 
                                  padding='post', truncating='post')
        
        # Get predictions
        svm_proba = self.svm_model.predict_proba(csv_scaled)[0, 1]
        rnn_proba = self.rnn_model.predict(nlp_padded, verbose=0)[0, 0]
        
        combined_proba = 0.4 * svm_proba + 0.6 * rnn_proba
        prediction = 1 if combined_proba >= 0.5 else 0
        
        if prediction == 1:
            confidence = combined_proba * 100
        else:
            confidence = (1 - combined_proba) * 100
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'svm_proba': svm_proba * 100,
            'rnn_proba': rnn_proba * 100,
            'combined_proba': combined_proba * 100
        }
        
    def display_results(self, result):
        """Display prediction results"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        is_fake = result['prediction'] == 1
        
        # Result card
        result_card = tk.Frame(self.result_frame, 
                              bg=self.colors['bg_card'])
        result_card.pack(fill='both', expand=True, padx=5, pady=10)
        
        # Main prediction
        prediction_frame = tk.Frame(result_card, 
                                    bg=self.colors['fake_color'] if is_fake else self.colors['real_color'])
        prediction_frame.pack(fill='x', padx=20, pady=20)
        
        emoji = "🚨" if is_fake else "✅"
        label = "FAKE ACCOUNT" if is_fake else "REAL ACCOUNT"
        
        tk.Label(prediction_frame, text=emoji,
                font=('Segoe UI', 48),
                bg=self.colors['fake_color'] if is_fake else self.colors['real_color'],
                fg=self.colors['text_primary']).pack(pady=(20, 10))
        
        tk.Label(prediction_frame, text=label,
                font=('Segoe UI', 24, 'bold'),
                bg=self.colors['fake_color'] if is_fake else self.colors['real_color'],
                fg=self.colors['text_primary']).pack()
        
        tk.Label(prediction_frame, 
                text=f"{result['confidence']:.1f}% Confidence",
                font=('Segoe UI', 16),
                bg=self.colors['fake_color'] if is_fake else self.colors['real_color'],
                fg=self.colors['text_primary']).pack(pady=(5, 20))
        
        # Detailed scores
        scores_frame = tk.Frame(result_card, bg=self.colors['bg_card'])
        scores_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(scores_frame, text="📊 Detailed Analysis",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(anchor='w', pady=(0, 10))
        
        self.create_score_bar(scores_frame, "Numerical Features (SVM)", 
                             result['svm_proba'])
        self.create_score_bar(scores_frame, "Text Features (RNN)", 
                             result['rnn_proba'])
        self.create_score_bar(scores_frame, "Combined Score", 
                             result['combined_proba'])
        
        # Recommendation
        rec_frame = tk.Frame(result_card, bg=self.colors['bg_secondary'])
        rec_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(rec_frame, text="💡 Recommendation",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_secondary']).pack(anchor='w', padx=15, pady=(15, 5))
        
        if is_fake:
            rec_text = "This account shows characteristics of a FAKE account.\n⚠️ Be cautious when interacting with this account."
        else:
            rec_text = "This account appears to be GENUINE.\n✓ Likely safe to interact with this account."
        
        tk.Label(rec_frame, text=rec_text,
                font=('Segoe UI', 10),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_secondary'],
                justify='left').pack(anchor='w', padx=15, pady=(0, 15))
        
    def create_score_bar(self, parent, label, score):
        """Create a score progress bar"""
        frame = tk.Frame(parent, bg=self.colors['bg_card'])
        frame.pack(fill='x', pady=8)
        
        # Label and score
        header = tk.Frame(frame, bg=self.colors['bg_card'])
        header.pack(fill='x')
        
        tk.Label(header, text=label,
                font=('Segoe UI', 9),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_card']).pack(side='left')
        
        tk.Label(header, text=f"{score:.1f}%",
                font=('Segoe UI', 9, 'bold'),
                fg=self.colors['accent'],
                bg=self.colors['bg_card']).pack(side='right')
        
        # Progress bar
        bar_bg = tk.Frame(frame, bg=self.colors['bg_secondary'], 
                         height=8)
        bar_bg.pack(fill='x', pady=(5, 0))
        
        bar_fill = tk.Frame(bar_bg, bg=self.colors['accent'], 
                           height=8)
        bar_fill.place(x=0, y=0, relwidth=score/100, relheight=1)
        
    def clear_fields(self):
        """Clear all input fields"""
        self.username_entry.delete(0, 'end')
        self.fullname_entry.delete(0, 'end')
        self.bio_entry.delete('1.0', 'end')
        self.posts_entry.delete(0, 'end')
        self.followers_entry.delete(0, 'end')
        self.following_entry.delete(0, 'end')
        self.captions_entry.delete('1.0', 'end')
        self.comments_entry.delete('1.0', 'end')
        
        self.profile_pic_var.set(True)
        self.external_url_var.set(False)
        self.private_var.set(False)
        
        # Clear results
        for widget in self.result_frame.winfo_children():
            widget.destroy()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ModernFakeAccountDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()