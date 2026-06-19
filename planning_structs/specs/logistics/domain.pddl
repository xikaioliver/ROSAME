(define (domain logistics-strips)
  (:requirements :strips :typing :equality)
  (:types
  	movable location city - object
  	obj transport - movable
  	truck airplane - transport
  	airport - location
  )
  (:predicates
		(at ?obj - movable ?loc - location)
		(in ?obj1 - obj ?obj2 - transport)
		(in-city ?loc - location ?city - city))
 
  ; (:types )		; default object

(:action load-truck
  :parameters
   (?obj - obj
    ?truck - truck
    ?loc - location)
  :precondition
   (and (at ?truck ?loc) (at ?obj ?loc))
  :effect
   (and (not (at ?obj ?loc)) (in ?obj ?truck)))

(:action load-airplane
  :parameters
   (?obj - obj
    ?airplane - airplane
    ?loc - airport)
  :precondition
   (and (at ?obj ?loc) (at ?airplane ?loc))
  :effect
   (and (not (at ?obj ?loc)) (in ?obj ?airplane)))

(:action unload-truck
  :parameters
   (?obj - obj
    ?truck - truck
    ?loc - location)
  :precondition
   (and (at ?truck ?loc) (in ?obj ?truck))
  :effect
   (and (not (in ?obj ?truck)) (at ?obj ?loc)))

(:action unload-airplane
  :parameters
   (?obj - obj
    ?airplane - airplane
    ?loc - airport)
  :precondition
   (and (in ?obj ?airplane) (at ?airplane ?loc))
  :effect
   (and (not (in ?obj ?airplane)) (at ?obj ?loc)))

(:action drive-truck
  :parameters
   (?truck - truck
    ?loc-from - location
    ?loc-to - location
    ?city - city)
  :precondition
   (and (at ?truck ?loc-from) (in-city ?loc-from ?city) (in-city ?loc-to ?city) (not (= ?loc-from ?loc-to)))
  :effect
   (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to)))

(:action fly-airplane
  :parameters
   (?airplane - airplane
    ?loc-from - airport
    ?loc-to - airport)
  :precondition
   (and (at ?airplane ?loc-from) (not (= ?loc-from ?loc-to)))
  :effect
   (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to)))
)
