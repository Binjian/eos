import tensorflow as tf
import unittest


def my_function(x):
    return x * 2


@tf.function
def decorated_function(x):
    input_signature = tf.TensorSpec(shape=[None], dtype=tf.float32)
    return my_function(x)


class TestDecoratedFunction(unittest.TestCase):
    def test_change_input_signature(self):
        # Change the input_signature to have a shape of [2]
        decorated_function.input_signature = tf.TensorSpec(shape=[2], dtype=tf.float32)

        # Call the function with a shape of [2]
        result = decorated_function([1, 2])

        # The result should be a tensor with a shape of [2]
        self.assertEqual(result.shape, [2])


if __name__ == "__main__":
    unittest.main()
