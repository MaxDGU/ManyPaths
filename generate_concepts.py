import numpy as np
import matplotlib.pyplot as plt
import random # Ensure random is imported for pcfg usage as well

from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept, DEFAULT_MAX_DEPTH as PCFG_DEFAULT_MAX_DEPTH

# --- Constants for example generation ---
EXAMPLE_POOL_SIZE = 256 # Number of random vectors to generate to find positive/negative examples
MAX_ATTEMPTS_FOR_SPECIFIC_EXAMPLE = 10 # Max attempts to find a specifically positive or negative example
                                      # before possibly re-sampling the concept or giving up.

def generate_concept(bits, scale: float = 1.0):
    if not (len(bits) == 4):
        raise ValueError("Bits must be length 4.")

    # Initialize a blank grid
    grid_image = np.ones((32, 32, 3), dtype=np.uint8) * 255

    # Extract tens and units digits
    color = (1, 2) if bits[0] == 1 else (0, 1)
    shape = bits[1] == 1
    size = 4 if bits[2] == 1 else 10
    style = bits[3] == 1

    if shape:
        grid_image[size : 32 - size, size : 32 - size, color] = 0
        if style == 1:
            grid_image[size : 32 - size, size : 32 - size : 2, color] = 200
    else:
        for i in range(32 - 2 * size):
            grid_image[
                32 - (size + i + 1), i // 2 + size : 32 - i // 2 - size, color
            ] = 0
        if style == 1:
            for i in range(0, 32, 1):
                for j in range(0, 32, 2):
                    if grid_image[i, j, color].any() == 0:
                        grid_image[i, j, color] = 200
    grid_image = grid_image / scale
    return grid_image


def plot():
    fig, axes = plt.subplots(2, 8, figsize=(14, 5))
    for i in range(16):
        bits = [int(x) for x in f"{i:04b}"]
        grid_image = generate_concept(bits)
        ax = axes[i // 8, i % 8]
        ax.imshow(grid_image, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{bits[0]}{bits[1]}{bits[2]}{bits[3]}", fontsize=20)

    plt.tight_layout(pad=0)
    output_path = "concept_grid.pdf"
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- PCFG-based Concept Generation ---
def define_random_pcfg_concept(num_features: int, max_depth: int = PCFG_DEFAULT_MAX_DEPTH) -> tuple[any, int, int]:
    """
    Defines a random concept using PCFG sampling.
    Returns the concept expression, its literal count, and its depth.
    """
    return sample_concept_from_pcfg(num_features, max_depth=max_depth)

def generate_example_for_pcfg_concept(
    concept_expression: any, 
    num_features: int, 
    force_positive: bool | None = None
) -> tuple[np.ndarray | None, int | None]:
    """
    Generates an example (input_vector, label) for a given PCFG concept_expression.
    Tries to find/generate examples that match the force_positive criteria.

    Args:
        concept_expression: The PCFG parse tree.
        num_features: Dimensionality of input vectors.
        force_positive: If True, try hard to generate a positive example.
                        If False, try hard to generate a negative example.
                        If None, generate randomly and return its label.

    Returns:
        A tuple (input_vector, label). Returns (None, None) if a specific type of example
        could not be found within attempts (especially for sparse positive concepts).
    """
    for attempt in range(MAX_ATTEMPTS_FOR_SPECIFIC_EXAMPLE):
        # Generate a pool of random candidate vectors
        # For force_positive=None, one random vector is enough for this attempt.
        # For True/False, we search within a pool.
        current_pool_size = EXAMPLE_POOL_SIZE if force_positive is not None else 1
        candidate_vectors = np.random.randint(0, 2, size=(current_pool_size, num_features), dtype=np.int8)
        
        for i in range(current_pool_size):
            input_vector = candidate_vectors[i]
            try:
                is_positive_eval = evaluate_pcfg_concept(concept_expression, input_vector)
            except Exception as e:
                # print(f"Warning: Error evaluating concept during example generation: {e}")
                # This might happen with ill-formed concepts or deep recursions if not handled by sampler
                continue # Try next candidate

            label = 1 if is_positive_eval else 0

            if force_positive is None:
                return input_vector, label # Return the first random one and its label
            elif force_positive is True and label == 1:
                return input_vector, 1 # Found a positive example
            elif force_positive is False and label == 0:
                return input_vector, 0 # Found a negative example
        
        # If after checking the pool, we haven't found the desired type for force_positive=True/False
        # and we are in such a mode, this attempt (outer loop) continues.
        # If force_positive is None, we would have returned already.

    # If max attempts reached and no suitable example found for force_positive=True/False
    # print(f"Warning: Could not generate {'positive' if force_positive else 'negative'} example for concept after {MAX_ATTEMPTS_FOR_SPECIFIC_EXAMPLE} attempts.")
    return None, None # Indicate failure to find specific example type


# OLD FUNCTIONS - Keep for reference or remove later
# def define_random_binary_concept(num_features: int) -> np.ndarray:
#     """
#     Defines a random binary concept as a target bit string.
#     Each bit represents a feature, and its value (0 or 1) is the required value for that feature.
#     Args:
#         num_features: The total number of binary features in the concept space.
#     Returns:
#         A numpy array of shape (num_features,) with binary values (0 or 1),
#         representing the target concept.
#     """
#     return np.random.randint(0, 2, size=num_features, dtype=np.int8)

# def generate_example_for_binary_concept(
#     target_concept_vector: np.ndarray, num_features: int = -1
# ) -> tuple[np.ndarray, int]:
#     """
#     Generates an example (input_vector, label) for a given binary target_concept_vector.
#     An input is a positive example (label 1) if it exactly matches the target_concept_vector.
#     Otherwise, it's a negative example (label 0).

#     Args:
#         target_concept_vector: A numpy array representing the target concept.
#         num_features: The dimensionality of the input vectors to generate.
#                       If -1 (default), it's inferred from target_concept_vector.
#                       It's good practice to provide it if known, otherwise it's inferred.

#     Returns:
#         A tuple containing:
#             - input_vector (np.ndarray): A random binary vector of shape (num_features,).
#             - label (int): 1 if input_vector matches target_concept_vector, 0 otherwise.
#     """
#     if num_features == -1:
#         num_features = len(target_concept_vector)
    
#     if len(target_concept_vector) != num_features:
#         raise ValueError(
#             f"target_concept_vector length ({len(target_concept_vector)}) "
#             f"must match num_features ({num_features})."
#         )

#     input_vector = np.random.randint(0, 2, size=num_features, dtype=np.int8)
#     label = 1 if np.array_equal(input_vector, target_concept_vector) else 0
#     return input_vector, label

# Example usage (can be run if this script is executed directly):
# if __name__ == "__main__":
#     # Test for new binary concept functions
#     num_concept_features = 8  # Example: 8 features for the concept
#     print(f"--- Testing Binary Concept Generation (Features: {num_concept_features}) ---")
#     current_target_concept = define_random_binary_concept(num_features=num_concept_features)
#     print(f"Target Concept: {current_target_concept}")

#     print("\\nGenerating 5 examples for this concept:")
#     for i in range(5):
#         example_input, example_label = generate_example_for_binary_concept(
#             current_target_concept, num_features=num_concept_features
#         )
#         print(f" Example {i+1}: Input={example_input}, Label={example_label}")
    
#     # Keep the original plot generation if needed, or comment out
#     # print("\\n--- Generating original visual concept grid ---")
#     # plot()


if __name__ == "__main__":
    # plot() # Original main execution retained
    
    test_num_features = 5 # Reduced for simpler testing of PCFG
    pcfg_max_depth_test = 3

    print(f"\\n--- Testing PCFG-Based Concept Generation (Features: {test_num_features}, Max Depth: {pcfg_max_depth_test}) ---")
    
    for i in range(3): # Sample a few concepts
        concept_expr, literals, depth = define_random_pcfg_concept(test_num_features, max_depth=pcfg_max_depth_test)
        print(f"\nConcept {i+1}: {concept_expr}")
        print(f"  Literals: {literals}, Depth: {depth}")

        print("  Attempting to generate examples:")
        # Try to get 2 positive examples
        print("    Positive examples:")
        for _ in range(2):
            ex_pos, lab_pos = generate_example_for_pcfg_concept(concept_expr, test_num_features, force_positive=True)
            if ex_pos is not None:
                print(f"      Input: {ex_pos}, Label: {lab_pos}, (Verified: {evaluate_pcfg_concept(concept_expr, ex_pos)})")
            else:
                print("      Failed to generate a positive example.")

        # Try to get 2 negative examples
        print("    Negative examples:")
        for _ in range(2):
            ex_neg, lab_neg = generate_example_for_pcfg_concept(concept_expr, test_num_features, force_positive=False)
            if ex_neg is not None:
                print(f"      Input: {ex_neg}, Label: {lab_neg}, (Verified: {not evaluate_pcfg_concept(concept_expr, ex_neg)})")
            else:
                print("      Failed to generate a negative example.")
        
        # Generate some random examples
        print("    Random examples:")
        for _ in range(3):
            ex_rand, lab_rand = generate_example_for_pcfg_concept(concept_expr, test_num_features, force_positive=None)
            if ex_rand is not None:
                 print(f"      Input: {ex_rand}, Label: {lab_rand}, (Verified label: {evaluate_pcfg_concept(concept_expr, ex_rand)})")
            else:
                 print("      Failed to generate random example (should not happen for force_positive=None unless error)." )
