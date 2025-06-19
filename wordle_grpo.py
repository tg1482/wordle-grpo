import re
import torch
import platform
import random
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


# Load words from file
def load_words_from_file(file_path: str = "wordle_words.txt") -> List[str]:
    """Load words from a text file. Supports both JSON array format and line-by-line format."""
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()

        # Try to parse as JSON first (for existing words_list.txt)
        if content.startswith("["):
            words = json.loads(content)
            return [word.upper().strip() for word in words]
        else:
            # Parse as line-by-line format
            words = [
                line.strip().upper() for line in content.split("\n") if line.strip()
            ]
            return words
    except FileNotFoundError:
        print(f"Words file {file_path} not found. Using fallback word list.")
        return ["HELLO", "WORLD", "TRAIN"]
    except Exception as e:
        print(f"Error loading words from {file_path}: {e}. Using fallback word list.")
        return ["HELLO", "WORLD", "TRAIN"]


# Load words at module level
WORD_LIST = load_words_from_file("words_list.txt")  # Try existing file first
if not WORD_LIST:
    WORD_LIST = load_words_from_file("wordle_words.txt")  # Then try new file
print(f"Loaded {len(WORD_LIST)} words for Wordle")


# Wordle Game Logic
class WordleGame:
    def __init__(self, target_word: str = None):
        self.target_word = target_word or self.sample_word()
        self.guesses = []
        self.feedback = []
        self.max_guesses = 6
        self.game_over = False
        self.won = False

    def sample_word(self) -> str:
        """Sample a random 5-letter word from the loaded word list."""
        return random.choice(WORD_LIST)

    def make_guess(self, guess: str) -> Tuple[List[str], bool]:
        """Make a guess and return feedback. Returns (feedback, game_won)"""
        if self.game_over:
            return self.feedback[-1] if self.feedback else [], self.won

        guess = guess.upper().strip()
        if len(guess) != 5:
            return ["INVALID"] * 5, False

        feedback = []
        target_chars = list(self.target_word)
        guess_chars = list(guess)

        # First pass: mark exact matches
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback.append("🟩")  # Green - correct letter, correct position
                target_chars[i] = None  # Mark as used
                guess_chars[i] = None  # Mark as used
            else:
                feedback.append("⬜")  # Placeholder

        # Second pass: mark partial matches
        for i in range(5):
            if guess_chars[i] is not None:  # Not already matched
                if guess_chars[i] in target_chars:
                    feedback[i] = "🟨"  # Yellow - correct letter, wrong position
                    target_chars[target_chars.index(guess_chars[i])] = None
                else:
                    feedback[i] = "⬛"  # Black - letter not in word

        self.guesses.append(guess)
        self.feedback.append(feedback)

        # Check win condition
        if guess == self.target_word:
            self.won = True
            self.game_over = True
        elif len(self.guesses) >= self.max_guesses:
            self.game_over = True

        return feedback, self.won

    def get_state(self) -> str:
        """Get current game state as a string."""
        state = f"Target: {'?' * 5 if not self.game_over else self.target_word}\n"
        state += f"Guesses remaining: {self.max_guesses - len(self.guesses)}\n\n"

        for i, (guess, feedback) in enumerate(zip(self.guesses, self.feedback)):
            state += f"Guess {i+1}: {guess} -> {''.join(feedback)}\n"

        if self.game_over:
            if self.won:
                state += f"\n🎉 You won in {len(self.guesses)} guesses!"
            else:
                state += f"\n💀 Game over! The word was {self.target_word}"

        return state

    def to_dict(self) -> Dict:
        """Convert game state to serializable dictionary."""
        return {
            "target_word": self.target_word,
            "guesses": self.guesses,
            "feedback": self.feedback,
            "max_guesses": self.max_guesses,
            "game_over": self.game_over,
            "won": self.won,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WordleGame":
        """Create game from serialized dictionary."""
        game = cls(target_word=data["target_word"])
        game.guesses = data["guesses"]
        game.feedback = data["feedback"]
        game.max_guesses = data["max_guesses"]
        game.game_over = data["game_over"]
        game.won = data["won"]
        return game


# Prompts and formatting
WORDLE_SYSTEM_PROMPT = """You are playing Wordle. You need to guess a 5-letter word.

For each guess, you'll see feedback:
🟩 = Correct letter in correct position
🟨 = Correct letter in wrong position  
⬛ = Letter not in the word

Respond in this format:
<reasoning>
Based on the feedback, I should...
</reasoning>
<guess>
YOURGUESS
</guess>"""


def extract_wordle_guess(text: str) -> str:
    """Extract the guess from the model's response."""
    try:
        guess = text.split("<guess>")[-1]
        guess = guess.split("</guess>")[0]
        return guess.strip().upper()
    except:
        # Fallback: try to find any 5-letter word
        words = re.findall(r"\b[A-Za-z]{5}\b", text)
        return words[0].upper() if words else "GUESS"


def create_wordle_dataset(num_games: int = 100) -> Dataset:
    """Create a dataset of Wordle game sessions."""
    games_data = []

    for _ in range(num_games):
        game = WordleGame()
        session_messages = [{"role": "system", "content": WORDLE_SYSTEM_PROMPT}]

        # Start the game
        session_messages.append(
            {
                "role": "user",
                "content": f"Let's play Wordle! You have 6 guesses to find the 5-letter word.\n\n{game.get_state()}\n\nMake your first guess:",
            }
        )

        games_data.append(
            {
                "prompt": session_messages.copy(),
                "target_word": game.target_word,
                "game_state": game.to_dict(),  # Store serializable dict instead of game object
                "guess_number": 1,
            }
        )

    return Dataset.from_list(games_data)


# Enhanced reward functions with output printing
def wordle_guess_quality_reward(
    prompts, completions, target_word, game_state, guess_number, **kwargs
) -> List[float]:
    """Reward based on the quality of the guess."""
    rewards = []

    for i, completion in enumerate(completions):
        try:
            response_content = completion[0]["content"]
            guess = extract_wordle_guess(response_content)

            # Print the model's output for debugging
            print(f"\n{'='*50}")
            print(
                f"SAMPLE {i+1} - Target: {target_word[0] if isinstance(target_word, list) else target_word}"
            )
            print(
                f"Guess #{guess_number[0] if isinstance(guess_number, list) else guess_number}"
            )
            print(f"Model Response:")
            print(f"{response_content}")
            print(f"Extracted Guess: {guess}")

            if len(guess) != 5 or not guess.isalpha():
                print(f"❌ Invalid guess format - Reward: -1.0")
                rewards.append(-1.0)
                continue

            # Check if guess is a valid word from our word list
            if guess not in WORD_LIST:
                print(f"❌ Not in word list - Reward: -0.5")
                rewards.append(-0.5)
                continue

            # Calculate letter accuracy
            target = target_word[0] if isinstance(target_word, list) else target_word
            correct_positions = sum(
                1
                for j in range(5)
                if j < len(guess) and j < len(target) and guess[j] == target[j]
            )
            correct_letters = sum(1 for letter in guess if letter in target)

            # Base reward for correct letters and positions
            reward = (
                correct_positions * 0.4 + (correct_letters - correct_positions) * 0.2
            )

            # Bonus for solving early
            if guess == target:
                early_bonus = max(
                    0,
                    2.0
                    - 0.3
                    * (
                        guess_number[0]
                        if isinstance(guess_number, list)
                        else guess_number - 1
                    ),
                )
                reward += early_bonus
                print(f"🎉 CORRECT GUESS! Early bonus: +{early_bonus:.2f}")

            # Small penalty for repeated letters if target doesn't have them
            unique_guess = len(set(guess))
            unique_target = len(set(target))
            if unique_guess < unique_target:
                reward -= 0.1

            print(f"📊 Correct positions: {correct_positions}/5")
            print(f"📊 Correct letters: {correct_letters}/5")
            print(f"🏆 Final reward: {reward:.2f}")

            rewards.append(reward)

        except Exception as e:
            print(f"❌ Error processing guess: {e}")
            print(f"Raw completion: {completion}")
            rewards.append(-0.5)

    return rewards


def wordle_format_reward(completions, **kwargs) -> List[float]:
    """Reward for following the correct format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<guess>.*?</guess>"
    rewards = []

    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        print(f"\n🔍 Format Check for Sample {i+1}:")

        if re.search(pattern, content, re.DOTALL):
            try:
                guess = extract_wordle_guess(content)
                if len(guess) == 5 and guess.isalpha() and guess in WORD_LIST:
                    print(f"✅ Perfect format + valid word - Reward: 0.5")
                    rewards.append(0.5)
                elif len(guess) == 5 and guess.isalpha():
                    print(f"⚠️ Good format but invalid word - Reward: 0.3")
                    rewards.append(0.3)
                else:
                    print(f"⚠️ Format ok but bad guess - Reward: 0.2")
                    rewards.append(0.2)
            except:
                print(f"⚠️ Format detected but extraction failed - Reward: 0.2")
                rewards.append(0.2)
        else:
            print(f"❌ Wrong format - Reward: 0.0")
            rewards.append(0.0)

    return rewards


def wordle_reasoning_quality_reward(completions, **kwargs) -> List[float]:
    """Reward for quality reasoning about the game state."""
    rewards = []

    for i, completion in enumerate(completions):
        content = completion[0]["content"].lower()
        reward = 0.0
        reasoning_points = []

        # Check for strategic thinking keywords
        if any(
            word in content
            for word in ["feedback", "eliminate", "narrow", "vowel", "consonant"]
        ):
            reward += 0.2
            reasoning_points.append("strategic thinking (+0.2)")

        if any(word in content for word in ["position", "correct", "wrong", "place"]):
            reward += 0.2
            reasoning_points.append("position awareness (+0.2)")

        if any(
            word in content for word in ["common", "frequent", "likely", "probable"]
        ):
            reward += 0.1
            reasoning_points.append("probability thinking (+0.1)")

        print(f"🧠 Reasoning Quality for Sample {i+1}:")
        if reasoning_points:
            for point in reasoning_points:
                print(f"   • {point}")
        else:
            print(f"   • No strategic reasoning detected")
        print(f"   💭 Reasoning reward: {reward:.2f}")

        rewards.append(reward)

    return rewards


# Custom training loop to show more details
class VerboseGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_counter = 0

    def compute_rewards(self, prompts, completions, **model_inputs):
        """Override to add logging."""
        self.step_counter += 1
        print(f"\n{'='*80}")
        print(f"TRAINING STEP {self.step_counter}")
        print(f"{'='*80}")

        # Print the prompt for context
        if prompts and len(prompts) > 0:
            print(f"\n📝 PROMPT:")
            for msg in prompts[0]:
                print(
                    f"{msg['role'].upper()}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}"
                )

        # Call the parent method to compute rewards
        rewards = super().compute_rewards(prompts, completions, **model_inputs)

        print(f"\n📊 TOTAL REWARDS SUMMARY:")
        if isinstance(rewards, list) and len(rewards) > 0:
            for i, reward_set in enumerate(rewards):
                if isinstance(reward_set, (list, tuple)) and len(reward_set) > 0:
                    total_reward = sum(reward_set)
                    print(f"   Sample {i+1}: {reward_set} = {total_reward:.3f}")

        print(f"\n{'='*80}\n")
        return rewards


# Device and model configuration
is_macos = platform.system() == "Darwin"
has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
has_cuda = torch.cuda.is_available()

if has_cuda:
    device = "cuda"
    torch_dtype = torch.bfloat16
    use_flash_attn = True
elif has_mps:
    device = "mps"
    torch_dtype = torch.float16
    use_flash_attn = False
else:
    device = "cpu"
    torch_dtype = torch.float32
    use_flash_attn = False

print(f"Using device: {device}")
print(f"Using dtype: {torch_dtype}")

# Model configuration
model_name = "Qwen/Qwen3-0.6B"
output_dir = "outputs/Qwen-0.6B-Wordle-GRPO"
run_name = "Qwen-0.6B-Wordle-GRPO"

# Create Wordle dataset
print("Creating Wordle dataset...")
wordle_dataset = create_wordle_dataset(num_games=10)  # Reduced for easier debugging
print(f"Created dataset with {len(wordle_dataset)} games")

# Print a sample from the dataset
print(f"\n📋 SAMPLE DATASET ENTRY:")
sample = wordle_dataset[0]
print(f"Target word: {sample['target_word']}")
print(f"Prompt: {sample['prompt'][-1]['content'][:300]}...")

# Training configuration
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=1e-7,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=device == "cuda",
    fp16=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Reduced for more frequent logging
    num_generations=2,
    max_prompt_length=512,
    max_completion_length=256,
    num_train_epochs=1,
    save_steps=25,  # More frequent saves
    max_grad_norm=1.0,
    report_to="wandb",
    log_on_each_node=False,
    temperature=0.8,
    top_p=0.9,
)

# PEFT configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

# Model loading
model_kwargs = {
    "torch_dtype": torch_dtype,
    "device_map": None,
}

if use_flash_attn:
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using flash attention")
    except Exception as e:
        print(f"Flash attention not available: {e}, using default attention")

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Create trainer with verbose output
trainer = VerboseGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        wordle_format_reward,
        wordle_reasoning_quality_reward,
        wordle_guess_quality_reward,
    ],
    args=training_args,
    train_dataset=wordle_dataset,
    # peft_config=peft_config  # Uncomment if you want to use PEFT
)

if __name__ == "__main__":
    print("Starting Wordle GRPO training...")
    print(f"Sample words from loaded list: {WORD_LIST[:10]}")

    # Test the model before training
    print(f"\n🧪 TESTING MODEL BEFORE TRAINING:")
    test_messages = [
        {"role": "system", "content": WORDLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Let's play Wordle! Make your first guess for a 5-letter word.",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        test_messages, return_tensors="pt", add_generation_prompt=True
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    print(f"Pre-training response: {response}")

    trainer.train()
    print("Training completed!")
