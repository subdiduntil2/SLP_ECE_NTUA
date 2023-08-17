EPS = "<eps>"  # Define once. Use the same EPS everywhere

CHARS = list("abcdefghijklmnopqrstuvwxyz")

INFINITY = 1000000000


def calculate_arc_weight(frequency):
    """Function to calculate the weight of an arc based on a frequency count

    Args:
        frequency (float): Frequency count

    Returns:
        (float) negative log of frequency

    """
    # TODO: INSERT YOUR CODE HERE
    raise NotImplementedError(
        "You need to implement calculate_arc_weight function in scripts/util.py!!!"
    )


def format_arc(src, dst, ilabel, olabel, weight=0):
     return f"{src} {dst} {ilabel} {olabel} {weight}"
    # return (str(src) + " " + str(dst) + " " + str(ilabel) + " " + str(olabel) + " " + str(weight) + "\n")
