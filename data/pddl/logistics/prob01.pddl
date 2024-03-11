(define (problem strips-log-x-1)
   (:domain logistics-strips)
   (:objects package6 package5 package4 package3 package2 package1 - OBJ
             city1 city2 - CITY
             truckred trucklime - TRUCK
             planeblue planeyellow - AIRPLANE
             city1-1 city2-1 - LOCATION
             city1-2 city2-2 - AIRPORT)
   (:init (in-city city2-2 city2)
          (in-city city2-1 city2)
          (in-city city1-2 city1)
          (in-city city1-1 city1)
          (at planeyellow city2-2)
          (at planeblue city1-2)
          (at trucklime city2-1)
          (at truckred city1-1)
          (at package6 city2-2)
          (at package5 city2-1)
          (at package4 city1-1)
          (in package3 truckred)
          (in package2 planeblue)
          (in package1 trucklime))
   (:goal (and (at package6 city1-2)
               (at package4 city2-2)
               (at package1 city2-1))))