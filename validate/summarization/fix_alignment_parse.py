import json


def parse_model_output(model_output_str):
    """
    Parse the model output JSON string and extract pred_labels and matched_lines.

    Args:
        model_output_str: A JSON string containing the model output

    Returns:
        A tuple containing (pred_labels, matched_lines)
    """
    pred_labels = []
    matched_lines = set()

    try:
        # Remove any markdown code block markers if present
        output = model_output_str.replace('```', '').strip()

        # Find the JSON part (looking for opening bracket)
        start_idx = output.find('[')
        if start_idx != -1:
            json_str = output[start_idx:]
            output_data = json.loads(json_str)

            for out in output_data:
                # Get the response category (yes/no)
                category = out.get("response", "").lower()
                pred_labels.append(1 if category == "yes" else 0)

                # Get matched line numbers
                line_nums = out.get("line number", [])
                for line_num in line_nums:
                    if isinstance(line_num, str):
                        # Clean string line numbers (remove brackets if present)
                        line_num = line_num.replace('[', '').replace(']', '').strip()
                    try:
                        matched_lines.add(int(line_num))
                    except ValueError:
                        continue

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error processing model output: {e}")

    return pred_labels, list(matched_lines)


# Example usage:
model_output_str = """[    {        "sentence": "The text provides an in-depth introduction to grey matter (GM) in the brain, focusing on its cellular composition, including neurons and glial cells, and their morphological characteristics.",        "quote": "Grey matter (GM) is composed of a range of cells, mainly differentiated into neuronal and glial cells, featuring a plethora of morphological characteristics.",        "category": "no error"    },    {        "sentence": "It highlights the historical study of neuronal morphology by Ramon y Cajal and the diversity of neuron types.",        "quote": "First studied in depth by Ramon y Cajal in the late 19th century [1], neuronal morphology offers insights into the complex structure and function of the brain. Neurons come in various shapes and sizes, tailored for their respective functions [2].",        "category": "no error"    },    {        "sentence": "The brain's GM is composed of cell bodies, neurites, extracellular space (ECS), and vasculature, with glial cells playing a crucial role in supporting neurons.",        "quote": "Cortical GM is roughly constituted of 10–40% cell bodies (soma) of neural cells; 40–75% neurites: neuronal dendrites, short-range intra-cortical axons, the stems of long-range axons extending into the WM and glial cell projections which intermingle with each other to form a dense and complex network; 15–30% highly tortuous extracellular space (ECS); and 1–5% vasculature [7, 8, 9].",        "category": "no error"    },    {        "sentence": "The ECS is described in terms of volume fraction, tortuosity, and width, with neuron-microvessel distance noted.",        "quote": "The ECS occupies a volume fraction of 15-30% in normal adult brain tissue, with a typical value of 20%, that falls to 5% during global ischemia (the expected state during classical fixation) [12]. The ECS has an average tortuosity (defined as the ratio between the true diffusion coefficient and the effective diffusivity of small molecules such as inulin and sucrose) of 2-3 [12], due to its labyrinthine porous matrix, the presence of long-chain macromolecules, transient trapping in dead-space microdomains, and transient physical-chemical interaction with the cellular membranes. The width of the ECS of the brain cortex is 40-120 nm [13]. The average neuron-microvessel distance in brain GM is 20 µm [15].",        "category": "no error"    },    {        "sentence": "The text discusses the limitations of current imaging techniques like MRI in visualizing cellular microstructure in vivo and introduces diffusion-weighted MRI (dMRI) as a promising alternative for characterizing brain structure at the cellular scale.",        "quote": "However, currently there are no means to directly observe the cellular microstructure in vivo and without invasively using imaging techniques, such as Magnetic Resonance Imaging (MRI), as the cellular scale (on micrometers) lies beyond the resolution of clinical MRI, generally on the mm scale [20]. Given its sensitivity to the micrometer length scale, diffusion-weighted MRI (dMRI) is a promising technique to address the resolution limit of MRI and characterize the brain structure in vivo at the cellular scale, that is, the microstructure.",        "category": "no error"    },    {        "sentence": "It reviews various dMRI-based biophysical models, such as DTI, NODDI, and WMTI, and their applications in white matter (WM) and GM.",        "quote": "Successful examples of the microstructure imaging paradigm include Diffusion Tensor Imaging (DTI) [22], the Neurite Orientation Dispersion and Density Imaging (NODDI) [23] and the White Matter Tract Integrity (WMTI) [24] to characterise the diffusion of water within white matter (WM), revealing insight into the structure of axonal bundle tracks and other anatomical features, such as axon diameter [25, 26], opening new and exciting opportunities in the field of neuroscience. Following the success of dMRI in WM, there has been a significant increase in interest in its application in GM, with the hopes of characterizing the cellular morphology within the brain in vivo.",        "category": "no error"    },    {        "sentence": "The text emphasizes the complexity of GM tissue and the challenges in developing biophysical models for GM microstructure imaging.",        "quote": "While substantial effort has been made to design, validate, and translate to clinics biophysical models for WM microstructure imaging [14, 31, 32], the GM counterpart is lagging. This is due to the greater complexity of the tissue, which renders the design of biophysical models for GM microstructure imaging more challenging [33].",        "category": "no error"    },    {        "sentence": "It concludes with the aim of providing a comprehensive analysis of GM cellular morphology using structural, topological, and shape descriptors from 3D reconstructions of brain cortex across different species, and offering guidelines for modeling dMRI signals in GM.",        "quote": "Here we aim to correct this imbalance with a comprehensive analysis of GM cellular morphology, looking at structural and topological morphology, and shape descriptors of over 3000 real three dimensional reconstructions from mouse, rat, monkey, and human brain cortex. We then review the range of dMRI measurements and biophysical models available to probe this anatomy and highlight limitations and caveats, ultimately providing guidelines on how to model dMRI signals in GM.",        "category": "no error"    }]"""

pred_labels, matched_lines = parse_model_output(model_output_str)
print(f"pred_labels: {pred_labels}")
print(f"matched_lines: {matched_lines}")