import pickle
import numpy as np
from lstm_numpy import LSTMRNN
from nlp_preprocessor import NLPPreprocessor

def main():
    print("=== Training Voice Assistant Intent Model ===")
    
    # 1. Prepare Data
    prep = NLPPreprocessor()
    prep.build_vocab('dataset.json')
    X, Y = prep.process_dataset('dataset.json')
    
    # Shuffle dataset
    dataset = list(zip(X, Y))
    np.random.shuffle(dataset)
    X, Y = zip(*dataset)
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Vocabulary size: {prep.vocab_size}")
    num_intents = len(prep.intent_to_idx)
    print(f"Number of intents: {num_intents}")
    
    # 2. Initialize Model
    # Input size is vocab_size (one-hot vectors), hidden size can be 32
    model = LSTMRNN(
        input_size=prep.vocab_size,
        hidden_size=32,
        output_size=num_intents,
        learning_rate=0.05
    )
    
    # 3. Train Model
    epochs = 150
    print(f"\nTraining LSTM on natural language dataset for {epochs} epochs...")
    model.train(X, Y, epochs=epochs)
    
    # 4. Evaluate (Simple accuracy on train set for demonstration)
    correct = 0
    for x_seq, y_target in zip(X, Y):
        pred = model.predict(x_seq)
        if pred == y_target:
            correct += 1
            
    accuracy = (correct / len(X)) * 100
    print(f"\nTraining Accuracy: {accuracy:.2f}%")
    if accuracy >= 85:
        print("Successfully achieved >= 85% accuracy on training sequences.")
    else:
        print("Accuracy is below 85%, you might want to train for more epochs.")
        
    # 5. Save model and preprocessor
    print("\nSaving model and preprocessor...")
    with open('intent_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(prep, f)
        
    print("Done! Model saved to intent_model.pkl")

if __name__ == '__main__':
    main()
