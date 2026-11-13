import json

def generate_notebook():
    cells = []
    
    # Cell 1: Markdown header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# The Liar's Chatroom - Winning Solution Notebook\n",
            "### Pure ML Approach (Open-Weight Reasoning LLM)\n",
            "- **Model**: `Qwen/Qwen2.5-14B-Instruct` (or `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`)\n",
            "- **Quantization**: 4-bit NF4 via `bitsandbytes` (fits on 1x T4 / P100 16GB GPU)\n",
            "- **Compliance**: 100% compliant with competition rules (No closed APIs, open-weight <70B, offline inference)\n",
            "- **Output**: Valid `submission.csv` (3,500 rows, matching test IDs)"
        ]
    })
    
    # Cell 2: Install dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q -U transformers accelerate bitsandbytes datasets"
        ]
    })
    
    # Cell 3: Code implementation
    with open('src/kaggle_qwen_reasoner.py', 'r', encoding='utf-8') as f:
        code_content = f.read()
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_content.splitlines()]
    })
    
    # Cell 4: Verification of submission
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import os\n",
            "\n",
            "sub_path = 'submission.csv' if os.path.exists('submission.csv') else 'submissions/submission.csv'\n",
            "df_sub = pd.read_csv(sub_path)\n",
            "print('Total rows:', len(df_sub))\n",
            "print('Header:', list(df_sub.columns))\n",
            "print('Null values:', df_sub.isnull().sum().to_dict())\n",
            "print('Label distribution:\\n', df_sub['label'].value_counts())\n",
            "df_sub.head(10)"
        ]
    })
    
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    out_file = 'kaggle_liars_chatroom_solution.ipynb'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated standalone notebook: {out_file}")

if __name__ == '__main__':
    generate_notebook()
