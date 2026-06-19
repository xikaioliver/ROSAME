(define (domain gripper-strips)
	(:requirements :adl :equality)
	(:types room ball gripper - object)
   (:predicates (at-robby ?r - room)
		(at ?b - ball ?r - room)
		(free ?g - gripper)
		(carry ?b - ball ?g - gripper))

   (:action move
       :parameters  (?from - room ?to - room)
       :precondition (and (at-robby ?from) (not (= ?from ?to)))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from))))


   (:action pick
       :parameters (?ball - ball ?room - room ?gripper - gripper)
       :precondition  (and (at ?ball ?room) (at-robby ?room) (free ?gripper))
       :effect (and (carry ?ball ?gripper)
		    (not (at ?ball ?room)) 
		    (not (free ?gripper))))


   (:action drop
       :parameters  (?ball - ball ?room - room ?gripper - gripper)
       :precondition  (and (carry ?ball ?gripper) (at-robby ?room))
       :effect (and (at ?ball ?room)
		    (free ?gripper)
		    (not (carry ?ball ?gripper)))))