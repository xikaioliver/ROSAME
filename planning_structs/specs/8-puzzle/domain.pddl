(define (domain strips-sliding-tile)
  (:requirements :strips :typing :equality)
  (:types
  	tile position - object
  )

  (:predicates
   (at ?t - tile ?p - position) (blank ?p - position)
   (neighbor ?p - position ?pp - position))

  (:action move
    :parameters (?t - tile ?from - position ?to - position)
    :precondition (and
		   (neighbor ?from ?to) (neighbor ?to ?from) (blank ?to) (at ?t ?from))
    :effect (and (not (blank ?to)) (not (at ?t ?from))
		 (blank ?from) (at ?t ?to)))

  )