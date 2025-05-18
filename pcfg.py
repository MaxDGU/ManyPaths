import random
import numpy as np

# --- Grammar Definition ---
# Non-Terminals: EXPR (can be an OR, AND, NOT, or a LITERAL)
# Terminals: Feature_i_True, Feature_i_False (dynamically handled)
# Structure: ('OPERATOR', child1, child2) or ('NOT', child) or 'Feature_i_True'/'Feature_i_False'

# Probabilities for production rules (must sum to 1 for alternatives from the same non-terminal)
P_EXPR_TO_OR = 0.25
P_EXPR_TO_AND = 0.25
P_EXPR_TO_NOT = 0.2
P_EXPR_TO_LITERAL = 0.3 # 1.0 - (P_EXPR_TO_OR + P_EXPR_TO_AND + P_EXPR_TO_NOT)

# Max depth for sampling to prevent infinite recursion
DEFAULT_MAX_DEPTH = 5

def sample_concept_from_pcfg(num_features: int, current_depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH) -> tuple[any, int, int]:
    """
    Samples a concept expression (parse tree) from the PCFG, and calculates its complexity.

    Args:
        num_features: Total number of available features.
        current_depth: Current recursion depth.
        max_depth: Maximum allowed recursion depth for sampling.

    Returns:
        A tuple: (expression, literals_count, expression_depth)
        - expression: The sampled concept, e.g., ('AND', 'F0_True', ('NOT', 'F1_False'))
        - literals_count: Number of feature literals in the expression.
        - expression_depth: Depth of the parse tree.
    """
    if current_depth >= max_depth:
        # Force a literal if max depth is reached
        # Choose a random feature and its state (True/False)
        feature_idx = random.randint(0, num_features - 1)
        is_true = random.choice([True, False])
        literal = f"F{feature_idx}_{'T' if is_true else 'F'}"
        return literal, 1, 1 # (expression, literals_count=1, depth=1)

    rand_val = random.random()

    if rand_val < P_EXPR_TO_OR:
        # EXPR -> (OR EXPR EXPR)
        child1, literals1, depth1 = sample_concept_from_pcfg(num_features, current_depth + 1, max_depth)
        child2, literals2, depth2 = sample_concept_from_pcfg(num_features, current_depth + 1, max_depth)
        return ('OR', child1, child2), literals1 + literals2, 1 + max(depth1, depth2)
    elif rand_val < P_EXPR_TO_OR + P_EXPR_TO_AND:
        # EXPR -> (AND EXPR EXPR)
        child1, literals1, depth1 = sample_concept_from_pcfg(num_features, current_depth + 1, max_depth)
        child2, literals2, depth2 = sample_concept_from_pcfg(num_features, current_depth + 1, max_depth)
        return ('AND', child1, child2), literals1 + literals2, 1 + max(depth1, depth2)
    elif rand_val < P_EXPR_TO_OR + P_EXPR_TO_AND + P_EXPR_TO_NOT:
        # EXPR -> (NOT EXPR)
        child, literals_child, depth_child = sample_concept_from_pcfg(num_features, current_depth + 1, max_depth)
        return ('NOT', child), literals_child, 1 + depth_child
    else:
        # EXPR -> LITERAL (forced, as it's the last option or if depth limit hit earlier)
        feature_idx = random.randint(0, num_features - 1)
        is_true = random.choice([True, False])
        literal = f"F{feature_idx}_{'T' if is_true else 'F'}"
        return literal, 1, 1


def evaluate_pcfg_concept(expression: any, input_vector: np.ndarray) -> bool:
    """
    Evaluates a PCFG-generated concept expression against a binary input vector.

    Args:
        expression: The concept expression (parse tree).
        input_vector: A numpy array of 0s and 1s representing feature values.

    Returns:
        True if the input_vector satisfies the concept, False otherwise.
    """
    if isinstance(expression, str): # It's a literal 'Fi_T' or 'Fi_F'
        parts = expression.split('_')
        feature_idx = int(parts[0][1:]) # Remove 'F'
        required_value_str = parts[1]

        if feature_idx >= len(input_vector):
            raise ValueError(f"Feature index {feature_idx} out of bounds for input_vector of length {len(input_vector)}.")

        actual_value = input_vector[feature_idx]
        if required_value_str == 'T':
            return actual_value == 1
        else: # 'F'
            return actual_value == 0
    
    op = expression[0]
    if op == 'NOT':
        return not evaluate_pcfg_concept(expression[1], input_vector)
    elif op == 'AND':
        return evaluate_pcfg_concept(expression[1], input_vector) and \
               evaluate_pcfg_concept(expression[2], input_vector)
    elif op == 'OR':
        return evaluate_pcfg_concept(expression[1], input_vector) or \
               evaluate_pcfg_concept(expression[2], input_vector)
    else:
        raise ValueError(f"Unknown operation in expression: {op}")

if __name__ == '__main__':
    test_num_features = 4
    print(f"--- Testing PCFG Concept Generation (Features: {test_num_features}, Max Depth: {DEFAULT_MAX_DEPTH}) ---")

    for i in range(5):
        concept_expr, literals, depth = sample_concept_from_pcfg(test_num_features, max_depth=3) # Test with a smaller depth
        print(f"\nSampled Concept {i+1}:")
        print(f"  Expression: {concept_expr}")
        print(f"  Literals: {literals}")
        print(f"  Depth: {depth}")

        # Test evaluation
        # Example: F0_T AND (NOT F1_F)
        # For input [1, 1, 0, 0]: F0 is T (1==1 True). F1 is T. (NOT F1_F) means (NOT (1==0)) = (NOT False) = True. So, T AND T = True.
        # For input [0, 0, 0, 0]: F0 is F. (F0_T) is False. Expression is False.
        
        test_input_1 = np.array([1, 1, 0, 0])
        test_input_2 = np.array([0, 0, 0, 0])
        
        try:
            eval_1 = evaluate_pcfg_concept(concept_expr, test_input_1)
            eval_2 = evaluate_pcfg_concept(concept_expr, test_input_2)
            print(f"  Eval on {test_input_1}: {eval_1}")
            print(f"  Eval on {test_input_2}: {eval_2}")
        except Exception as e:
            print(f"  Error during evaluation: {e}")

    print("\n--- Testing a specific complex expression ---")
    # ('OR', ('AND', 'F0_T', 'F1_F'), ('NOT', ('AND', 'F2_T', 'F3_F')))
    # Literals = 4, Depth = 3
    # F0=T, F1=F, F2=T, F3=F
    complex_expr = ('OR', ('AND', 'F0_T', 'F1_F'), ('NOT', ('AND', 'F2_T', 'F3_F')))
    print(f"Expression: {complex_expr}")
    # Input1: [1,0,1,0] -> (AND T T) OR (NOT (AND T T)) -> T OR (NOT T) -> T OR F -> TRUE
    # Input2: [1,0,0,0] -> (AND T T) OR (NOT (AND F T)) -> T OR (NOT F) -> T OR T -> TRUE
    # Input3: [0,1,1,0] -> (AND F F) OR (NOT (AND T T)) -> F OR (NOT T) -> F OR F -> FALSE
    
    inputs_for_complex = [
        np.array([1,0,1,0]), # Expected: True
        np.array([1,0,0,0]), # Expected: True
        np.array([0,1,1,0])  # Expected: False
    ]
    expected_outputs = [True, True, False]

    for i, t_inp in enumerate(inputs_for_complex):
        res = evaluate_pcfg_concept(complex_expr, t_inp)
        print(f"  Eval on {t_inp}: {res} (Expected: {expected_outputs[i]})")
        assert res == expected_outputs[i]
    print("Specific complex expression test passed.") 