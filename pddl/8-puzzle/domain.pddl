(define (domain strips-sliding-tile)
  (:requirements :strips :typing :equality)
  (:types
  	tile position - object
  )

  (:predicates
   (at ?p - position ?t - tile) (blank ?p - position)
   (neighbor ?p - position ?pp - position))

  (:action move
    :parameters (?from - position ?to - position ?t - tile)
    :precondition (and
		   (neighbor ?from ?to) (neighbor ?to ?from) (blank ?to) (at ?from ?t))
    :effect (and (not (blank ?to)) (not (at ?from ?t))
		 (blank ?from) (at ?to ?t)))

  )