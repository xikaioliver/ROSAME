(define (domain logistics-strips)
  (:requirements :strips :typing :equality)
  (:types
  	movable location city - object
  	obj transport - movable
  	truck airplane - transport
  	airport - location
  )
  (:predicates
		(at ?loc - location ?obj - movable)
		(in ?obj1 - obj ?obj2 - transport)
		(in-city ?city - city ?loc - location))
 
  ; (:types )		; default object

(:action load-truck
  :parameters
   (?loc - location
    ?obj - obj
    ?truck - truck)
  :precondition
   (and (at ?loc ?truck) (at ?loc ?obj))
  :effect
   (and (not (at ?loc ?obj)) (in ?obj ?truck)))

(:action load-airplane
  :parameters
   (?airplane - airplane
    ?loc - airport
    ?obj - obj)
  :precondition
   (and (at ?loc ?obj) (at ?loc ?airplane))
  :effect
   (and (not (at ?loc ?obj)) (in ?obj ?airplane)))

(:action unload-truck
  :parameters
   (?loc - location
    ?obj - obj
    ?truck - truck)
  :precondition
   (and (at ?loc ?truck) (in ?obj ?truck))
  :effect
   (and (not (in ?obj ?truck)) (at ?loc ?obj)))

(:action unload-airplane
  :parameters
   (?airplane - airplane
    ?loc - airport
    ?obj - obj)
  :precondition
   (and (in ?obj ?airplane) (at ?loc ?airplane))
  :effect
   (and (not (in ?obj ?airplane)) (at ?loc ?obj)))

(:action drive-truck
  :parameters
   (?city - city
    ?loc-from - location
    ?loc-to - location
    ?truck - truck)
  :precondition
   (and (at ?loc-from ?truck) (in-city ?city ?loc-from) (in-city ?city ?loc-to) (not (= ?loc-from ?loc-to)))
  :effect
   (and (not (at ?loc-from ?truck)) (at ?loc-to ?truck)))

(:action fly-airplane
  :parameters
   (?airplane - airplane
    ?loc-from - airport
    ?loc-to - airport)
  :precondition
   (and (at ?loc-from ?airplane) (not (= ?loc-from ?loc-to)))
  :effect
   (and (not (at ?loc-from ?airplane)) (at ?loc-to ?airplane)))
)
