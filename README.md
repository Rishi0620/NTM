#Neural Turing Machine – Learning Algorithmic Tasks

A PyTorch implementation of a **Neural Turing Machine (NTM)** – a neural network architecture augmented with differentiable memory and attention mechanisms – designed to learn algorithmic tasks such as **sorting, copying, reversing**, and **duplicating** sequences.

Built and trained locally on a MacBook Air (M2, CPU), this project demonstrates how NTMs can learn simple programs through data alone.

---

##Project Goals

- Implement a fully differentiable Neural Turing Machine from scratch
- Train the model on algorithmic sequence transformation tasks
- Explore task generalization using a shared model with task-conditioning
- Run efficiently on local CPU hardware (M2 MacBook Air)

---

## Model Architecture

- **Controller:** LSTM
- **Memory:** External differentiable memory matrix
- **Attention Mechanism:** Content-based addressing for read/write heads
- **Differentiable Operations:** Soft attention for memory read/write

---

## Tasks Trained

| Task        | Description                            | Input Example                | Output Example              |
|-------------|----------------------------------------|------------------------------|-----------------------------|
| Sorting     | Sort float sequences in ascending order| `[0.3, 0.1, 0.2]`            | `[0.1, 0.2, 0.3]`           |
| Copying     | Output input as-is                     | `[0.5, 0.8]`                 | `[0.5, 0.8]`                |
| Reversing   | Reverse the input sequence             | `[0.4, 0.6, 0.7]`            | `[0.7, 0.6, 0.4]`           |
| Duplicating | Repeat the input sequence              | `[0.2, 0.9]`                 | `[0.2, 0.9, 0.2, 0.9]`      |

Multi-task learning was implemented using a **task token**, allowing a single model to learn multiple tasks with generalization.

---

###Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib (for visualization)

###Installation

```bash
git clone https://github.com/yourusername/neural-turing-machine
cd neural-turing-machine
pip install -r requirements.txt
