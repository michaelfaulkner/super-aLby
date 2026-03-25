from c_imp_get_next_event_folder.c_imp_get_next_event import get_next_event

event_test = c_imp_get_next_event.get_next_event(10,1,100,1.37,1.37,1.37,0.5,0.5,1,1.0,-10.0,1.5)

print(event_test.shortest_distance_to_next_event)