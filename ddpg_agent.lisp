;;; Version Notes: renamed to starcraft from starcrat
;;; A1rev3: Added a second production that /should/ repond to imaginal bufferings.  It doesn't fire yet.
;;; A1rev2: Modified to receive a response
;;; - reduced the number of cycles to 1000
;;; A1rev1: just a loop model. Does not receive a response
;;; from starcraft_imaginal to starcraft_learn




;; Define the ACT-R model. This will generate a pair of warnings about slots in the goal
;; buffer not being accessed from productions; these may be ignored.

(clear-all)
(require-extra "blending")
(define-model ddpg


(sgp :esc t    ;;set to t to use subsymbolic computations
     :sim-hook "cosine-similarity"
     :act nil;t 
     ;:cache-sim-hook-results t
     ;:bln t    ;;added this to ensure blending was enabled, but not sure if it is actually doing anything...
     ;;some parameters moved from +actr-parameters+ in swarm model
     :ans nil;0.25 ;;activation noise, default is nil, swarm was .75 (M-turkers), Christian recommended .25 as a start.
     :tmp nil  ;;decouple noise if not nil, swarm was .7, perhaps change later.
     :mp 1   ;;partial matching enabled, default is off/nil, start high (2.5 is really high) and move to 1.
     :bll nil;0.5  ;;base-level learning enabled, default is off, this is the recommended value.
     :rt -10.0 ;;retrieval threshold, default is 0, think about editing this number.
     :blc 5;5    ;;base-level learning constant, default is 0, think about editing this number.
     :lf 0.25  ;;latency factor for retrieval times, default is 1.0, think about editing this number.
     :ol nil   ;;affects which base-level learning equation is used, default is nil; use 1 later maybe
     :md -2.5  ;;maximum difference, default similarity value between chunks, default is 1.0, maybe edit this.
     :ga 0.0   ;;set goal activation to 0.0 for spreading activation
     ;:imaginal-activation 1.0 ;;set imaginal activation to 0.0 for spreading activation
     ;:mas 1.0 ;;spreading activation
     ;;back to some parameters in the original sgp here
     :v t      ;;set to nil to turn off model output
     :blt nil;t    ;;set to nil to turn off blending trace
     :trace-detail low;high ;;lower this as needed, start at high for initial debugging.
     :style-warnings t  ;;set to nil to turn off production warnings, start at t for initial debugging.
     ) ;;end sgp


(chunk-type initialize state)
;(chunk-type observation state)
(chunk-type decision green_x green_y orange_x orange_y player_x player_y
green_north green_north_east green_east green_south_east green_south green_south_west green_west green_north_west
orange_north orange_north_east orange_east orange_south_east orange_south orange_south_west orange_west orange_north_west
action reward)
(chunk-type action value)
(chunk-type reward value)
(chunk-type game_state extreme_left far_left left ahead right far_right extreme_right
 distance_to_target angle_to_target relative_alt wait)
;(chunk-type green isa color value green)
;(chunk-tupe orange isa color value orange)



(add-dm
 (goal isa initialize state setup)
 (select-orange isa action value select-orange)
 (select-green isa action value select-green)
 ;(select-beacon isa action value select-beacon)
 (select-around isa action value select-around))
;;chunks defined in Python and vector have random elements

(P initial_production
   =goal>
       isa        initialize
       state      setup
   =imaginal>
     - wait       true
   ==>
   =goal>
       state      observe
   =imaginal>
       wait       true

 
   
   !eval! ("tic")
   ;!eval! (format t "msg")
)

(P end
   =imaginal>
       stop       true
==>
   -imaginal>
   -goal>
)

(P wait
   =imaginal>
       wait       true
   ?imaginal>
       state      free
   ==>
   =imaginal>

       wait       true

   !eval! ("tic")
   
)
;;should the wait production tic as well?
;;my initial thought is no because it 
;;should be happening between tics




(P make_observation
   =goal>
       state      observe
   =imaginal>
       isa        game_state
       extreme_left =exLeft
       far_left   =farLeft
       left       =left
       ahead      =ahead
       right      =right
       far_right  =farRight
       extreme_right =exRight
       distance_to_target   =distance
       angle_to_target      =angle
       relative_alt =relAlt
    ; - wait       true
   ?retrieval>
       state      free
       buffer     empty
     - state      error
==>
   ;+retrieval>
       ;isa        decision ;notice the change from game-state to decision
       ;green      =green
       ;orange     =orangexs
       ;between    =between
       ;vector     =vector
       ;value_estimate =value_estimate
       ;:ignore-slots (wait)
     ;- action     nil
       ;:do-not-generalize (green orange between vector value_estimate)
   =imaginal>
       extreme_left nil
       far_left   nil
       left       nil
       ahead      nil
       right      nil
       far_right  nil
       extreme_right nil
       distance_to_target   nil
       angle_to_target      nil
       relative_alt nil
       wait       false
   =goal>
       state      setup

   !eval! ("set_response" 0 =angle)
)







;;;;;;;;;;



(P click_north
   =goal>
       isa        initialize
       state      click_direction
   =imaginal>
       player_x   =p_x
       player_y   =p_y
==>

   !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" "north" p_x p_y)
   =imaginal>
       stop        true
)



(P get_action
   =goal>
       isa        initialize
       state      get_action
   =retrieval>
       isa        decision
       green      =green_r
       orange     =orange_r
       between    =between_r
       vector     =vector_r
       value_estimate =value_estimate_r
       action     =action_r
   =imaginal>
       isa        decision
       green      =green
       orange     =orange
       between    =between
       vector     =vector
       value_estimate =value_estimate

   
==>
  
   =goal>
       state      store_instance
   @retrieval>
   @imaginal>
       isa        decision
       green      =green
       orange     =orange
       between    =between
       vector     =vector
       value_estimate =value_estimate
       action     =action_r
       ;wait       nil

   ;!eval! ("Blend")

)

(P store_instance
  =goal>
      state        store_instance
  =imaginal>
      isa          decision
      green        =green
      orange       =orange
      between      =between
      vector       =vector
      value_estimate =value_estimate
      action       =action
==>
  =goal>
      state        do_action
  =imaginal>

  !eval! ("Blend")
)

(P select-green
   =goal>
       state      do_action
   =imaginal>
       action     select-green
     - wait       true

==>
    =goal>
       state      check-neutrals
    =imaginal>
       wait       true
       action     nil
    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_green")
)

(P select-orange
   =goal>
       state      do_action
   =imaginal>
       action     select-orange
     - wait       true
==>
    =goal>
       state      check-neutrals
    =imaginal>
       wait       true
       action     nil
    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_orange")
)

(P select-beacon
   =goal>
       state      do_action
   =imaginal>
       action     select-beacon
     - wait       true
==>
    =goal>
       state      check-neutrals
    =imaginal>
       wait       true
       action     nil
    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_beacon")
)
(P select-around
   =goal>
       state      do_action
   =imaginal>
       action     select-around
     - wait       true
==>
    =goal>
       state      check-neutrals
    =imaginal>
       wait       true
       action     nil
    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_around")
)
     

(goal-focus goal)
) ; end define-model

