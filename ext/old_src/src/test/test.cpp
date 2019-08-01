#define CATCH_CONFIG_MAIN
#include "catch.hpp"


SCENARIO ( "normal running", "[main]" ) {

    GIVEN ( "anything" ) {

        WHEN ( "anything" ) {

            THEN ("the result is correct") {

                REQUIRE(0 == 0);

            }

        }
    
    }

}
