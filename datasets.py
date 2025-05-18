import os
import torch
import random
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
from torchvision.datasets.utils import list_files
from collections import defaultdict
from torch.utils.data import Dataset
from generate_numbers import generate_number_grid
from generate_concepts import generate_concept # For visual concepts if still used
from generate_concepts import define_random_pcfg_concept, generate_example_for_pcfg_concept, PCFG_DEFAULT_MAX_DEPTH
from constants import FEATURE_VALUES

MAX_CONCEPT_RESAMPLES = 10 # Max times to try resampling a concept if example generation fails
CACHE_DIR = "data/concept_cache" # Directory for caching generated tasks

class BaseMetaDataset(Dataset):
    def __init__(self):
        self.tasks = []

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task_data = self.tasks[idx]
        # Adjusted to match the extended task tuple including complexity metrics
        return task_data[0], task_data[2], task_data[3], task_data[5], task_data[6], task_data[7], task_data[8]

    def _image_to_patches(self, image_batch, patch_size=4):
        B, C, H, W = image_batch.shape
        assert H % patch_size == 0 and W % patch_size == 0
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.reshape(B, C, -1, patch_size, patch_size).permute(
            0, 2, 1, 3, 4
        )
        return patches.reshape(B, -1, C * patch_size * patch_size)


class MetaBitConceptsDataset(BaseMetaDataset):
    def __init__(
        self,
        n_tasks: int = 10000,
        data: str = "bits", # Should always be "bits" for this class now
        model: str = "mlp",  # Model type, affects input processing sometimes
        n_support: int = None, # If provided, used to set k_shot_pos/neg for eval
        num_features: int = 8, 
        pcfg_max_depth: int = PCFG_DEFAULT_MAX_DEPTH,
        k_shot_positive: int = 2,
        k_shot_negative: int = 3,
        n_query_positive: int = 5,
        n_query_negative: int = 5
    ):
        super().__init__()
        assert data == "bits", "MetaBitConceptsDataset is designed for 'bits' data type only."
        self.n_tasks = n_tasks
        self.model = model # Mainly for potential model-specific input processing, less relevant for MLP bits
        self.num_features = num_features
        self.pcfg_max_depth = pcfg_max_depth
        
        self.k_shot_positive_target = k_shot_positive
        self.k_shot_negative_target = k_shot_negative
        self.n_query_positive_target = n_query_positive
        self.n_query_negative_target = n_query_negative

        # If n_support is provided (e.g. for evaluation from init_dataset),
        # override the k_shot_positive/negative for this instance.
        if n_support is not None:
            self.k_shot_positive_target = max(1, n_support // 2)
            self.k_shot_negative_target = max(1, n_support - self.k_shot_positive_target)
            # Also scale query examples if n_support is given (e.g. for comparable eval task sizes)
            self.n_query_positive_target = max(1, n_support) # e.g. n_support pos query examples
            self.n_query_negative_target = max(1, n_support) # e.g. n_support neg query examples
        
        # --- Task Caching Logic ---
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Construct a filename based on key parameters that define the dataset uniqueness
        cache_filename_parts = [
            "pcfg_tasks",
            f"f{self.num_features}",
            f"d{self.pcfg_max_depth}",
            f"s{self.k_shot_positive_target}p{self.k_shot_negative_target}n",
            f"q{self.n_query_positive_target}p{self.n_query_negative_target}n",
            f"t{self.n_tasks}"
        ]
        self.cache_file = os.path.join(CACHE_DIR, "_".join(cache_filename_parts) + ".pt")

        if os.path.exists(self.cache_file):
            print(f"Loading PCFG concept tasks from cache: {self.cache_file}")
            try:
                self.tasks = torch.load(self.cache_file, weights_only=True)
            except RuntimeError as e: # PyTorch often raises RuntimeError for weights_only issues
                print(f"Warning: Failed to load cache with weights_only=True ({e}). This might happen if the cache was saved with an older PyTorch version or contains non-tensor data. Falling back to weights_only=False.")
                self.tasks = torch.load(self.cache_file, weights_only=False) # This will emit FutureWarning
            
            if len(self.tasks) != self.n_tasks:
                print(f"Warning: Cached tasks count ({len(self.tasks)}) doesn't match requested n_tasks ({self.n_tasks}). Regenerating.")
                self._generate_tasks_and_cache()
            else:
                 print(f"Successfully loaded {len(self.tasks)} tasks from cache.")
        else:
            print(f"No cache found at {self.cache_file}. Generating PCFG concept tasks...")
            self._generate_tasks_and_cache()

    def _generate_tasks_and_cache(self):
        self._generate_tasks() # This populates self.tasks
        if self.tasks: # Only save if tasks were actually generated
            print(f"Saving {len(self.tasks)} generated PCFG tasks to cache: {self.cache_file}")
            try:
                torch.save(self.tasks, self.cache_file)
                print("Tasks successfully saved to cache.")
            except Exception as e:
                print(f"Error saving tasks to cache: {e}")
        else:
            print("No tasks generated, cache not saved.")

    def _generate_tasks(self):
        # This method is for bit tasks using PCFG generated concepts
        self.tasks = [] # Ensure tasks list is clear before starting generation
        tasks_generated = 0
        while tasks_generated < self.n_tasks:
            current_concept_expression = None
            concept_literals, concept_depth = -1, -1 # Default/error values
            
            support_inputs_list = []
            support_labels_list = []
            query_inputs_list = []
            query_labels_list = []
            
            valid_task_generated = False
            for concept_resample_attempt in range(MAX_CONCEPT_RESAMPLES):
                concept_expression, concept_literals, concept_depth = define_random_pcfg_concept(
                    self.num_features, max_depth=self.pcfg_max_depth
                )
                current_concept_expression = concept_expression # Store for potential reuse if only some examples fail

                # Clear lists for new concept attempt
                support_inputs_list.clear()
                support_labels_list.clear()
                query_inputs_list.clear()
                query_labels_list.clear()
                
                possible_to_generate = True

                # Generate positive support examples
                for _ in range(self.k_shot_positive_target):
                    inp, lab = generate_example_for_pcfg_concept(concept_expression, self.num_features, force_positive=True)
                    if inp is None: possible_to_generate = False; break
                    support_inputs_list.append(inp)
                    support_labels_list.append(lab)
                if not possible_to_generate: continue # Resample concept

                # Generate negative support examples
                for _ in range(self.k_shot_negative_target):
                    inp, lab = generate_example_for_pcfg_concept(concept_expression, self.num_features, force_positive=False)
                    if inp is None: possible_to_generate = False; break
                    support_inputs_list.append(inp)
                    support_labels_list.append(lab)
                if not possible_to_generate: continue # Resample concept

                # Generate positive query examples
                for _ in range(self.n_query_positive_target):
                    inp, lab = generate_example_for_pcfg_concept(concept_expression, self.num_features, force_positive=True)
                    if inp is None: possible_to_generate = False; break
                    query_inputs_list.append(inp)
                    query_labels_list.append(lab)
                if not possible_to_generate: continue # Resample concept

                # Generate negative query examples
                for _ in range(self.n_query_negative_target):
                    inp, lab = generate_example_for_pcfg_concept(concept_expression, self.num_features, force_positive=False)
                    if inp is None: possible_to_generate = False; break
                    query_inputs_list.append(inp)
                    query_labels_list.append(lab)
                if not possible_to_generate: continue # Resample concept
                
                valid_task_generated = True # All examples generated successfully
                break # Break from concept_resample_attempt loop
            
            if not valid_task_generated:
                print(f"Warning: Failed to generate a valid task after {MAX_CONCEPT_RESAMPLES} concept resamples. Skipping task {tasks_generated + 1}.")
                # Optionally, could raise an error or have a fallback (e.g. simpler concept)
                # For now, we just might generate fewer than n_tasks if this happens often.
                # To strictly meet n_tasks, this loop structure might need adjustment or error handling.
                continue # Try to generate the next task for the while loop

            # Shuffle support and query sets independently
            support_combined = list(zip(support_inputs_list, support_labels_list))
            random.shuffle(support_combined)
            s_inputs_shuffled, s_labels_shuffled = zip(*support_combined) if support_combined else ([], [])

            query_combined = list(zip(query_inputs_list, query_labels_list))
            random.shuffle(query_combined)
            q_inputs_shuffled, q_labels_shuffled = zip(*query_combined) if query_combined else ([], [])

            X_s_numpy = np.array(s_inputs_shuffled, dtype=np.float32) if s_inputs_shuffled else np.empty((0,self.num_features), dtype=np.float32)
            y_s_numpy = np.array(s_labels_shuffled, dtype=np.float32).reshape(-1, 1) if s_labels_shuffled else np.empty((0,1), dtype=np.float32)
            X_q_numpy = np.array(q_inputs_shuffled, dtype=np.float32) if q_inputs_shuffled else np.empty((0,self.num_features), dtype=np.float32)
            y_q_numpy = np.array(q_labels_shuffled, dtype=np.float32).reshape(-1, 1) if q_labels_shuffled else np.empty((0,1), dtype=np.float32)

            X_s_original_tensor = torch.from_numpy(X_s_numpy)
            y_s_tensor = torch.from_numpy(y_s_numpy)
            X_q_original_tensor = torch.from_numpy(X_q_numpy)
            y_q_tensor = torch.from_numpy(y_q_numpy)

            X_s_processed_tensor = X_s_original_tensor * 2.0 - 1.0 # Scale to {-1, 1}
            X_q_processed_tensor = X_q_original_tensor * 2.0 - 1.0
            
            if self.model in ["lstm", "transformer"]:
                X_s_processed_tensor = X_s_processed_tensor.unsqueeze(2)
                X_q_processed_tensor = X_q_processed_tensor.unsqueeze(2)

            self.tasks.append((
                X_s_processed_tensor, X_s_original_tensor, y_s_tensor,
                X_q_processed_tensor, X_q_original_tensor, y_q_tensor,
                len(s_inputs_shuffled), # Actual number of support examples
                concept_literals,       # PCFG concept_literals
                concept_depth           # PCFG concept_depth
            ))
            tasks_generated += 1
            if tasks_generated % 1000 == 0:
                 print(f"Generated {tasks_generated}/{self.n_tasks} PCFG concept tasks...")

    def _generate_image_tasks(self):
        # This is the old method for image-based concepts. Keep if needed for other experiments.
        # For now, it will not be called if data="bits" is asserted in __init__.
        # If you need to support both, the dispatch in _generate_tasks would be needed.
        from grammer import DNFHypothesis # Import DNFHypothesis here as it's only used for image tasks now
        X_q_orig_bits = torch.tensor(FEATURE_VALUES, dtype=torch.float) # These are the 4-bit patterns
        X_image_q = torch.zeros((16, 3, 32, 32))
        for i, bits_arr in enumerate(FEATURE_VALUES): # bits_arr is a list/array of 4 bits
            grid_image = generate_concept(bits_arr, scale=255.0).reshape(3, 32, 32) # generate_concept expects list/array
            X_image_q[i] = torch.from_numpy(grid_image)
        
        mean = X_image_q.mean(dim=[0, 2, 3])
        std = X_image_q.std(dim=[0, 2, 3])
        X_image_q_processed = (X_image_q - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        if self.model == "mlp":
            X_image_q_processed = X_image_q_processed.reshape(16, 32 * 32 * 3)
        elif self.model in ["lstm", "transformer"]:
            X_image_q_processed = self._image_to_patches(X_image_q_processed)
        
        while len(self.tasks) < self.n_tasks: # This should be tasks_generated < self.n_tasks for consistency
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels_for_feature_values = [hyp.function(f_val) for f_val in FEATURE_VALUES] 
            if len(set(labels_for_feature_values)) < 2: continue

            n_support_current = (
                np.random.randint(2, high=20) 
                if getattr(self, 'k_shot_positive_target', None) is None # Crude check if n_support wasn't used
                else self.k_shot_positive_target + self.k_shot_negative_target # Use combined target
            )
            if n_support_current == 0 : n_support_current = 1 

            inds = np.random.choice(16, size=(n_support_current,), replace=True) 
            X_s_orig_bits_current = X_q_orig_bits[inds]
            X_image_s_processed_current = X_image_q_processed[inds]
            y_s_current = torch.tensor(labels_for_feature_values, dtype=torch.float).unsqueeze(1)[inds]
            y_q_current = torch.tensor(labels_for_feature_values, dtype=torch.float).unsqueeze(1) 
            
            # For image tasks, concept_literals and concept_depth are not directly applicable from PCFG
            # We can add placeholders or derive them differently if needed.
            concept_literals_img, concept_depth_img = -1, -1 # Placeholder for image tasks

            self.tasks.append((
                X_image_s_processed_current, X_s_orig_bits_current, y_s_current, 
                X_image_q_processed, X_q_orig_bits, y_q_current, n_support_current,
                concept_literals_img, concept_depth_img # Add placeholders
            ))
            # tasks_generated += 1 # Increment if this loop is the main one for image tasks


class MetaModuloDataset(BaseMetaDataset):
    def __init__(
        self,
        n_tasks=10000,
        n_moduli=20,
        range_max=100,
        skip=1,
        train=True,
        sigma=0.1,
        data="image",
        model="cnn",
        bit_width=8,
        n_support: int = None,  # for test-time, testing across n_support #
    ):
        super().__init__()
        assert skip in [1, 2]
        assert data in ["image", "bits", "number"]
        self.n_tasks = n_tasks
        self.n_moduli = n_moduli
        self.range_max = range_max
        self.skip = skip
        self.train = train
        self.sigma = sigma
        self.data = data
        self.model = model
        self.bit_width = bit_width
        self.n_support = n_support
        self._generate_tasks()

    def _generate_tasks(self):
        offset = 0 if self.train else (self.n_moduli if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        moduli = list(range(1 + offset, mul * self.n_moduli + 1 + offset, self.skip))
        if self.n_tasks > self.n_moduli:
            ms = torch.tensor(
                [moduli[i] for i in torch.randint(0, len(moduli), (self.n_tasks,))]
            )
        else:
            ms = moduli
        for m in ms:
            if self.data == "image":
                self.tasks.append(self._generate_image_task(m))
            elif self.data == "bits":
                self.tasks.append(self._generate_bits_task(m))
            else:
                self.tasks.append(self._generate_number_task(m))

    def _generate_image_task(self, m):
        n_support = (
            np.random.randint(10, high=101)
            if self.n_support is None
            else self.n_support
        )
        X_s = torch.randint(0, self.range_max, (n_support, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()
        X_image_q = torch.zeros((self.range_max, 1, 32, 32))
        for i, num in enumerate(X_q[:, 0].int().numpy()):
            grid_image = generate_number_grid(num).reshape(1, 32, 32)
            X_image_q[i] = torch.from_numpy(grid_image)
        mean = X_image_q.mean(dim=[0, 2, 3])
        std = X_image_q.std(dim=[0, 2, 3])
        X_image_q = (X_image_q - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        X_image_s = X_image_q[X_s[:, 0].int()]
        if self.model == "mlp":
            X_image_s = X_image_s.view(n_support, 32 * 32)
            X_image_q = X_image_q.view(self.range_max, 32 * 32)
        elif self.model in ["lstm", "transformer"]:
            X_image_s = self._image_to_patches(X_image_s)
            X_image_q = self._image_to_patches(X_image_q)
        return (X_image_s, X_s, y_s, X_image_q, X_q, y_q, m)

    def _generate_bits_task(self, m):
        n_support = (
            np.random.randint(10, high=101)
            if self.n_support is None
            else self.n_support
        )
        X_s = torch.randint(0, self.range_max, (n_support, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()
        X_bits_s = self._generate_bitstrings(X_s.int())
        X_bits_q = self._generate_bitstrings(X_q.int())
        if self.model == "mlp":
            X_bits_s = X_bits_s.squeeze()
            X_bits_q = X_bits_q.squeeze()
        return (X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, n_support)

    def _generate_number_task(self, m):
        n_support = (
            np.random.randint(10, high=101)
            if self.n_support is None
            else self.n_support
        )
        X_s = torch.randint(0, self.range_max, (n_support, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()
        return (X_s, X_s, y_s, X_q, X_s, y_q, m)

    def _generate_bitstrings(self, numbers):
        b = torch.arange(self.bit_width - 1, -1, -1, dtype=torch.int32)
        bitstrings = (numbers >> b) & 1
        return (bitstrings * 2 - 1).unsqueeze(-1).float()


class Omniglot(BaseMetaDataset):
    DATA_PATH = "./omniglot"

    def __init__(
        self,
        n_tasks: int = 10000,
        alphabet: list = None,
        model: str = "cnn",
        N: int = 20,
        K: int = 5,
        train=True,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.model = model
        self.N = N
        self.K = K
        self.train = train
        self.alphabet = alphabet
        self.transform_train = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 32)),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.922059], std=[0.268076]),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.922059], std=[0.268076]),
            ]
        )
        self.transform = self.transform_train if self.train else self.transform_test
        self.dataset, self.characters, self.image_cache = self._init_dataset()
        self._generate_tasks()

    def _init_dataset(self):
        raw_dataset = datasets.Omniglot(
            root=self.DATA_PATH, background=self.train, download=False # change to true if you don't have the dataset
        )
        self.target_folder = raw_dataset.target_folder
        images_per_char = []
        for character in raw_dataset._characters:
            char_path = os.path.join(self.target_folder, character)
            # list_files returns all *.png in that character directory
            images_per_char.append(
                [
                    (file_name,) + os.path.split(character)
                    for file_name in list_files(char_path, ".png")
                ]
            )
        dataset = defaultdict(lambda: defaultdict(list))
        for group in images_per_char:
            for file_name, alpha, character in group:
                dataset[alpha][character].append(file_name)
        # Filter by specified alphabets (if provided and if training)
        if self.alphabet is not None and self.train:
            dataset = {a: dataset[a] for a in self.alphabet if a in dataset}
        characters = [(a, c) for a in dataset for c in dataset[a]]

        image_cache = {}
        for alpha in dataset:
            for char in dataset[alpha]:
                for img_name in dataset[alpha][char]:
                    # Full path
                    img_path = os.path.join(self.target_folder, alpha, char, img_name)
                    # Load once, store the raw PIL image
                    # (We apply transforms later, so random transformations still differ each time.)
                    image_cache[img_path] = self.transform(Image.open(img_path).convert("L"))

        return dataset, characters, image_cache

    def _generate_tasks(self):
        for _ in range(self.n_tasks):
            characters = random.sample(self.characters, self.N)
            X_s, X_q, y_s, y_q = [], [], [], []
            for i, (alphabet, character) in enumerate(characters):
                images = self.dataset[alphabet][character]
                np.random.shuffle(images)
                support = [os.path.join(self.target_folder, alphabet, character, img) for img in images[: self.K]]
                query = [os.path.join(self.target_folder, alphabet, character, img) for img in images[self.K :]]
                X_s.extend(support)
                y_s.extend([i] * self.K)  # Class label `i` for support set
                X_q.extend(query)
                y_q.extend([i] * len(query))  # Class label `i` for query set
            X_s_imgs = [self.image_cache[path] for path in X_s]
            X_q_imgs = [self.image_cache[path] for path in X_q]
            X_s = torch.stack(X_s_imgs)
            X_q = torch.stack(X_q_imgs)
            if self.model == "mlp":
                X_s = X_s.view(X_s.size(0), -1)  
                X_q = X_q.view(X_q.size(0), -1)
            elif self.model in ["lstm", "transformer"]:
                X_s = self._image_to_patches(X_s)
                X_q = self._image_to_patches(X_q)
            y_s = torch.tensor(y_s, dtype=torch.long)
            y_q = torch.tensor(y_q, dtype=torch.long)
            self.tasks.append((X_s, None, y_s, X_q, None, y_q, None))
