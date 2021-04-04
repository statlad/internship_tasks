class LinAlg:
    """
    Basic linear algebra program.
    Contains methods for addition, subtraction, element division, element multiplication,
    matrix multiplication and determinant.
    """

    def __init__(self, ):
        self.mx = None

    def shape(self, mx=None):

        if mx is None:
            mx = self.mx
        return len(mx), len(mx[0])

    def shape_check(self, a, b):
        """
        Shape check of the matrices
        Parameters
        ----------
        a : n-dim array
        b : n-dim array

        Returns
        -------

        """

        if self.shape(a)[0] != self.shape(b)[0] or self.shape(a)[1] != self.shape(b)[1]:
            raise ValueError('Matrices are not same shape')

        return True

    def add(self, a, b):
        """
        Element addition of matrix A and B
        Parameters
        ----------
        a : n-dim array
        b : n-dim array

        Returns
        -------
        n-dim array
        """

        if self.shape_check(a, b):
            shape_a = self.shape(a)
            shape_b = self.shape(b)

            sum_matrix = [[None for _ in range(shape_a[1])] for _ in range(shape_b[0])]
            for i in range(shape_a[0]):
                for j in range(shape_b[1]):
                    sum_matrix[i][j] = a[i][j] + b[i][j]

            return sum_matrix

    def sub(self, a, b):
        """
        Element subtraction of matrix A and B
        Parameters
        ----------
        a : n-dim array
        b : n-dim array

        Returns
        -------
        n-dim array
        """

        if self.shape_check(a, b):
            shape_a = self.shape(a)
            shape_b = self.shape(b)

            sub_matrix = [[None for _ in range(shape_a[1])] for _ in range(shape_b[0])]
            for i in range(shape_a[0]):
                for j in range(shape_b[1]):
                    sub_matrix[i][j] = a[i][j] - b[i][j]

            return sub_matrix

    def div(self, a, b):
        """
        Element division of matrix A and B
        Parameters
        ----------
        a : n-dim array
        b : n-dim array

        Returns
        -------
        n-dim array
        """

        if self.shape_check(a, b):
            shape_a = self.shape(a)
            shape_b = self.shape(b)

            div_matrix = [[None for _ in range(shape_a[1])] for _ in range(shape_b[0])]
            for i in range(shape_a[0]):
                for j in range(shape_b[1]):
                    div_matrix[i][j] = a[i][j] / b[i][j]

            return div_matrix

    def elem_multiply(self, a, b):
        """
        Element multiplication of matrix A and B
        Parameters
        ----------
        a : n-dim array
        b : n-dim array

        Returns
        -------
        n-dim array
        """

        if self.shape_check(a, b):
            shape_a = self.shape(a)
            shape_b = self.shape(b)

            mult_matrix: list = [[None for _ in range(shape_a[1])] for _ in range(shape_b[0])]
            for i in range(shape_a[0]):
                for j in range(shape_b[1]):
                    mult_matrix[i][j] = a[i][j] * b[i][j]

            return mult_matrix

    def transpose(self, a):
        """
        Matrix transposition
        Parameters
        ----------
        a : n-dim array

        Returns
        -------
        n-dim array
        """

        shape_a = self.shape(a)

        m_t: list = [[None for _ in range(shape_a[0])] for _ in range(shape_a[1])]
        for i in range(shape_a[0]):
            for j in range(shape_a[1]):
                m_t[j][i] = a[i][j]
        return m_t

    def determinant(self, a, mul=1):
        """
        Calculates determinant of matrix
        Parameters
        ----------
        a : n-dim array
        mul : int

        Returns
        -------
        float
        """

        if self.shape(a)[0] != self.shape(a)[1]:
            raise ValueError('Matrix is not square')

        if self.shape(a)[0] == 1:
            return mul * a[0][0]

        width = len(a)

        if width == 1:
            return mul * a[0][0]
        else:
            sign = -1
            sum_i = 0

            for i in range(width):
                m = []
                for j in range(1, width):
                    buff = []
                    for k in range(width):
                        if k != i:
                            buff.append(a[j][k])
                    m.append(buff)
                sign *= -1
                sum_i += mul * self.determinant(m, sign * a[0][i])

            return sum_i

    def multiply(self, a, b):
        """
        Matrix multiplication A with B.
        Parameters
        ----------
        a : MxN matrix
        b : NxK matrix

        Returns
        -------
        MxK matrix
        """

        if self.shape(a)[1] != self.shape(b)[0]:
            raise ValueError('Shapes do not match')

        mat_mul: list = [[None for _ in range(self.shape(b)[1])] for _ in range(self.shape(a)[0])]
        for i in range(self.shape(a)[0]):
            for j in range(self.shape(b)[1]):
                total = 0
                for ii in range(self.shape(a)[1]):
                    total += a[i][ii] * b[ii][j]
                mat_mul[i][j] = total

        return mat_mul


if __name__ == '__main__':
    linalg = LinAlg()

