(define (domain hanoi)
  (:requirements :strips :typing :equality)
  (:types
  	disc - object
  )
  (:predicates
  (clear ?x - object)
  (on ?x - disc ?y - object)
  (smaller ?x - disc ?y - object)
  )

  (:action move
    :parameters (?d - disc ?from - object ?to - object)
    :precondition (and (smaller ?d ?to) (on ?d ?from)
               (clear ?d) (clear ?to))
    :effect  (and (clear ?from) (on ?d ?to) (not (on ?d ?from))
          (not (clear ?to))))
  )