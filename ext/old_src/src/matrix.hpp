

namespace cai {

    class Matrix {

        float* values;
        int row_factor, column_factor;

        public:
            int rows, columns;

            Matrix(int rows, int columns, bool x_indexed);

            void set(int row, int column, float value);
            void set_all(float values[]);
            void fill_rand(float range_a, float range_b);

            float get(int row, int column);

            void print();
    };

    void matrix_multiply(Matrix *a, Matrix *b, Matrix *c);
    void matrix_add(Matrix *a, Matrix *b, Matrix *c);
    void matrix_rectify(Matrix *a, float (*f)(float, float), float alpha);

    float matrix_diff(Matrix *a, Matrix *b);

};