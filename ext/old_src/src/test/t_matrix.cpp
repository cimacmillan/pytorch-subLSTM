#include "matrix.hpp"
#include "catch.hpp"

SCENARIO ( "matrices are correct", "[matrix]" ) {

    GIVEN ( "2x2 matrices" ) {

        cai::Matrix a(2, 2, true);
        cai::Matrix a2(2, 2, false);
        cai::Matrix b(2, 2, true);
        cai::Matrix c(2, 2, true);

        float array_a[] = {1, 2, 3, 4};
        float array_b[] = {5, 6, 7, 8};

        a.set_all(array_a);
        a2.set_all(array_a);
        b.set_all(array_b);

        WHEN ( "they are multiplied with normal index" ) { //Section

            cai::matrix_multiply(&a, &b, &c);

            THEN ("the result is correct") {

                REQUIRE( c.get(0, 0) == 19 );
                REQUIRE( c.get(0, 1) == 22 );
                REQUIRE( c.get(1, 0) == 43 );
                REQUIRE( c.get(1, 1) == 50 );

            }
        }

        WHEN ( "they are multiplied with abnormal index" ) { //Section

            cai::matrix_multiply(&a2, &b, &c);

            THEN ("the result is correct") {

                REQUIRE( c.get(0, 0) == 19 );
                REQUIRE( c.get(0, 1) == 22 );
                REQUIRE( c.get(1, 0) == 43 );
                REQUIRE( c.get(1, 1) == 50 );

            }
        }
    }

}



