import json
import threading
import socket
import time

current_connection = None

class request():
    def __init__(self,id):
        self.id = id
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.complete = False

    def notify_result(self):
        self.cv.acquire()
        self.complete = True
        self.cv.notify()
        self.cv.release()


locals = threading.local()

class actr():
    
    
    def __init__(self,host="127.0.0.1",port=2650):
        self.interface = interface(host, port)
        self.interface.echo_output()

    def evaluate (self, *params):
        
        try:
            m = locals.model_name
        except AttributeError:
            m = False
     
        p = list(params)

        p.insert(1,m)    

        r = self.interface.send ("evaluate", *p)
        
        if r[0] == False:
            for e in r[1:]:
                print (e)

            return False
        else:
            return r[1:]

    def evaluate_single(self,*params):
        r = self.evaluate(*params)

        if r:
            return r[0]
        else:
            return False

    def add_command(self,name,function,documentation="No documentation provided.",single=True):
        if name not in self.interface.commands.keys():
            self.interface.add_command(name,function)
        elif self.interface.commands[name] == function:
            print("Command ",name," already exists for function ",function)
        else:
            print("Command ",name," already exists and is now being replaced by ",function)
            self.interface.add_command(name,function)
 
        existing = self.interface.send("check",name)

        if existing[1] == None:
            self.interface.send("add",name,name,documentation,single)
        elif existing[2] == None:
            print("Cannot add command ",name, " because it has already been added by a different owner.")
            return False
        
        return True

    def monitor_command(self,original,monitor):
        r = self.interface.send("monitor",original,monitor)

        if r[0] == False:
            for e in r[1:]:
                print (e)

            return False
        else:
            return r[1:]

 
    def remove_command_monitor(self,original,monitor):
        r = self.interface.send("remove-monitor",original,monitor)

        if r[0] == False:
            for e in r[1:]:
                print (e)

            return False
        else:
            return r[1:]       

    def remove_command(self,name):
        if name not in self.interface.commands.keys():
            print("Command ",name," does not exist to be removed.")
            return False
        else:
            del self.interface.commands[name]
            r = self.interface.send("remove",name)
            
            if r[0] == False:
                for e in r[1:]:
                    print (e)

                return False
            else:
                return True



def start (host="127.0.0.1",port=2650):

    global current_connection

    if current_connection == None: 
        current_connection = actr(host, port)
        return current_connection
    else:
        print("ACT-R is already connected.")
        return current_connection

def connection ():

    if current_connection == None:
        print("ACT-R connection has been started with default parameters.")
        return start()
    else:
        return current_connection

def stop():

    global current_connection

    if current_connection == None:
        print("No current ACT-R connection to stop.")
    else:
        print("Closing down ACT-R connection.")
        current_connection.interface.connected = False
        current_connection.interface.sock.close()
        current_connection = None


class interface():
    def __init__(self,host="127.0.0.1",port=2650):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.connected = True
        self.cmd_id = 1
        self.actions = {}
        self.stream_lock = threading.Lock() 
        self.buffer = []
        self.commands = {}
        self.data_collector = threading.Thread(target=self.collect_data)
        self.data_collector.daemon = True
        self.data_collector.start()       
        self.id_lock = threading.Lock()
        self.echo_count = 0
        self.echo = False
        self.show_output = True

    def send(self,method,*params):
        d = {}
        r = request(self.cmd_id)
        self.actions[self.cmd_id] = r

        d['method'] = method
        self.id_lock.acquire()
        d['id'] = self.cmd_id
        self.cmd_id += 1
        self.id_lock.release()
        d['params'] = params
        
        message = json.dumps(d) + chr(4)
        
        r.lock.acquire()
        
        self.stream_lock.acquire()
        self.sock.sendall(message.encode('utf-8'))
        self.stream_lock.release()
        
        while not r.complete:
          r.cv.wait()

        return [r.success] + r.results


    def add_command(self,name,function):
        self.commands[name] = function

    def collect_data(self):
        buffer= ''
        c = True
        while c:
            try:
                data = self.sock.recv(4096)
                buffer += data.decode('utf-8')
                while not chr(4) in buffer:
                    data = self.sock.recv(4096)
                    buffer += data.decode('utf-8')
                while chr(4) in buffer:
                    pos = buffer.find(chr(4))
                    message = buffer[0:pos]
                    pos += 1
                    buffer = buffer[pos:]
                    self.process_message(json.loads(message))
            except:
                if self.connected:
                    print("ACT-R connection error connection no longer available.")
                c = False

    def process_message (self,d):
        if 'result' in d.keys():
            id =d['id']
            r = self.actions[id]
            if d['error'] is None:
                r.success = True
                r.results = d['result']
            else:
                r.success = False
                errors=d['error']
                r.results = [errors['message']]

            self.actions.pop(id,None)
            r.notify_result()
        else:
            if d['method'] == "evaluate" and d['params'][0] in self.commands.keys():
                thread = threading.Thread(target=self.run_command,args=[self.commands[d['params'][0]],d['params'][1],d['id'],d['params'][2:]])
                thread.daemon = True
                thread.start()
            else:
                f={}
                f['id'] = d['id']
                f['result'] = None
                e={}
                e['message'] = "Invalid method name" + d['params'][0]
                f['error'] = e
                message = json.dumps(f) + chr(4)
                self.stream_lock.acquire()
                self.sock.sendall(message.encode('utf-8'))
                self.stream_lock.release()

    def run_command (self,command,model,id,params):

        locals.model_name = model

        if params == None:
            result = command()
        else:
            result = command(*params)
        f={}
        f['id'] = id
        if result:
            f['result']= result
        else:
            f['result']= [None]
        f['error']= None
        message = json.dumps(f) + chr(4)
        self.stream_lock.acquire()
        self.sock.sendall(message.encode('utf-8'))
        self.stream_lock.release()
        
    def output_monitor(self,string):
        if self.show_output:
            print(string.rstrip())
        return True

    def echo_output(self):
        if not(self.echo): 
            if 'echo' not in self.commands.keys():
                self.add_command("echo",self.output_monitor)

            ready = False

            while not(ready):
                existing = self.send("check",'python-echo'+str(self.echo_count))

                if existing[1] == None:
                    self.send("add","python-echo"+str(self.echo_count),"echo","Trace monitor for python client.  Do not call directly.",True)
                    ready = True
                else:
                    self.echo_count += 1

        
            self.send("monitor","model-trace","python-echo"+str(self.echo_count))
            self.send("monitor","command-trace","python-echo"+str(self.echo_count))
            self.send("monitor","warning-trace","python-echo"+str(self.echo_count))
            self.send("monitor","general-trace","python-echo"+str(self.echo_count))
            self.echo = True
            return True

        else:
            print("echo_output called when output was already on.")
            return False

    def no_output(self):
    
        if self.echo:
            self.send("remove-monitor","model-trace","python-echo"+str(self.echo_count))
            self.send("remove-monitor","command-trace","python-echo"+str(self.echo_count))
            self.send("remove-monitor","warning-trace","python-echo"+str(self.echo_count))
            self.send("remove-monitor","general-trace","python-echo"+str(self.echo_count))
            self.send("remove","python-echo"+str(self.echo_count))
            self.echo = False
        else:
            print("no_output called when output was already off.")

current_connection = connection()

def reset ():
    return current_connection.evaluate_single("reset")

def reload (compile=False):
    return current_connection.evaluate_single("reload",compile)

def run (time, real_time=False):
    return current_connection.evaluate("run", time, real_time)

def run_full_time (time, real_time=False):
    return current_connection.evaluate("run-full-time", time, real_time)

def run_until_time (time, real_time=False):
    return current_connection.evaluate("run-until-time", time, real_time)

def run_n_events (event_count, real_time=False):
    return current_connection.evaluate("run-n-events", event_count, real_time)

def run_until_condition(condition,real_time=False):
    return current_connection.evaluate("run-until-condition", condition, real_time)

def buffer_chunk (*params):
    return current_connection.evaluate_single("buffer-chunk", *params)

def whynot (*params):
    return current_connection.evaluate_single("whynot", *params)

def whynot_dm (*params):
    return current_connection.evaluate_single("whynot-dm", *params)


def penable (*params):
    return current_connection.evaluate_single("penable", *params)

def pdisable (*params):
    return current_connection.evaluate_single("pdisable", *params)

def load_act_r_model (path):
    return current_connection.evaluate_single("load-act-r-model",path)

def goal_focus (goal=None):
    return current_connection.evaluate_single("goal-focus",goal)

def clear_exp_window(win=None):
    return current_connection.evaluate_single("clear-exp-window",win)


def open_exp_window(title,visible=True,width=300,height=300,x=300,y=300):
    return current_connection.evaluate_single("open-exp-window",title,visible,width,height,x,y)

def add_text_to_exp_window(window,text,x=0,y=0,color='black',height=20,width=75,font_size=12):
    return current_connection.evaluate_single("add-text-to-exp-window",window,text,x,y,color,height,width,font_size)

def add_button_to_exp_window(window,text,x=0,y=0,action=None,height=20,width=75,color='gray'):
    return current_connection.evaluate_single("add-button-to-exp-window",window,text,x,y,action,height,width,color)

def remove_items_from_exp_window(window,*items):
    return current_connection.evaluate_single("remove-items-from-exp-window",window,*items)


def install_device(device):
    return current_connection.evaluate_single("install-device",device)

def print_warning(warning):
    current_connection.evaluate("print-warning",warning)

def act_r_output(output):
    current_connection.evaluate("act-r-output",output)

def random(value):
    return current_connection.evaluate_single("act-r-random",value)


def add_command(name,function,documentation="No documentation provided.",single=True):
    return current_connection.add_command(name,function,documentation,single)

def monitor_command(original,monitor):
    return current_connection.monitor_command(original,monitor)
 
def remove_command_monitor(original,monitor):
    return current_connection.remove_command_monitor(original,monitor)

def remove_command(name):
    return current_connection.remove_command(name)

def print_visicon():
    return current_connection.evaluate_single("print-visicon")

def mean_deviation(results,data,output=True):
    return current_connection.evaluate_single("mean-deviation",results,data,output)

def correlation(results,data,output=True):
    return current_connection.evaluate_single("correlation",results,data,output)

def get_time(model_time=True):
    return current_connection.evaluate_single("get-time",model_time)

def buffer_status (*params):
    return current_connection.evaluate_single("buffer-status", *params)

def buffer_read (buffer):
    return current_connection.evaluate_single("buffer-read", buffer)

def clear_buffer (buffer):
    return current_connection.evaluate_single("clear-buffer", buffer)

def new_tone_sound (freq, duration, onset=False, time_in_ms=False):
    return current_connection.evaluate_single("new-tone-sound", freq, duration, onset, time_in_ms)

def new_word_sound (word, onset=False, location='external', time_in_ms=False):
    return current_connection.evaluate_single("new-word-sound", word, onset, location, time_in_ms)

def new_digit_sound (digit, onset=False, time_in_ms=False):
    return current_connection.evaluate_single("new-digit-sound", digit, onset, time_in_ms)

def define_chunks (*chunks):
    return current_connection.evaluate_single("define-chunks", *chunks)

def define_chunks_fct (chunks):
    return current_connection.evaluate_single("define-chunks", *chunks)

def add_dm (*chunks):
    return current_connection.evaluate_single("add-dm", *chunks)

def add_dm_fct (chunks):
    return current_connection.evaluate_single("add-dm-fct", chunks)

def pprint_chunks (*chunks):
    return current_connection.evaluate_single("pprint-chunks", *chunks)

def chunk_slot_value (chunk_name, slot_name):
    return current_connection.evaluate_single("chunk-slot-value", chunk_name, slot_name)

def set_chunk_slot_value (chunk_name, slot_name, new_value):
    return current_connection.evaluate_single("set-chunk-slot-value", chunk_name, slot_name, new_value)

def mod_chunk (chunk_name, *mods):
    return current_connection.evaluate_single("mod-chunk", chunk_name, *mods)

def mod_focus (*mods):
    return current_connection.evaluate_single("mod-focus", *mods)

def chunk_p (chunk_name):
    return current_connection.evaluate_single("chunk-p",chunk_name)

def copy_chunk (chunk_name):
    return current_connection.evaluate_single("copy-chunk",chunk_name)

def extend_possible_slots (slot_name, warn=True):
    return current_connection.evaluate_single("extend-possible-slots",slot_name,warn)

def model_output (output_string):
    return current_connection.evaluate_single("model-output",output_string)


def set_buffer_chunk (buffer_name, chunk_name, requested=True):
    return current_connection.evaluate_single("set-buffer-chunk",buffer_name,chunk_name,requested)

def add_line_to_exp_window (window, start, end, color = False):
    if color:
        return current_connection.evaluate_single("add-line-to-exp-window",window,start,end,color)
    else:
        return current_connection.evaluate_single("add-line-to-exp-window",window,start,end)

def modify_line_for_exp_window (line, start, end, color = False):
    if color:
        return current_connection.evaluate_single("modify-line-for-exp-window",line,start,end,color)
    else:
        return current_connection.evaluate_single("modify-line-for-exp-window",line,start,end)

def start_hand_at_mouse ():
    return current_connection.evaluate_single("start-hand-at-mouse")

def schedule_simple_event (time, action, params=None, module='NONE', priority=0, maintenance=False):
    return current_connection.evaluate_single("schedule-simple-event",time,action,params,module,priority,maintenance)

def schedule_simple_event_now (action, params=None, module='NONE', priority=0, maintenance=False):
    return current_connection.evaluate_single("schedule-simple-event-now",action,params,module,priority,maintenance)

def schedule_simple_event_relative (time_delay, action, params=None, module='NONE', priority=0, maintenance=False):
    return current_connection.evaluate_single("schedule-simple-event-relative",time_delay,action,params,module,priority,maintenance)

def mp_show_queue(indicate_traced=False):
    return current_connection.evaluate_single("mp-show-queue",indicate_traced)

def print_dm_finsts():
    return current_connection.evaluate_single("print-dm-finsts")

def spp (*params):
    return current_connection.evaluate_single("spp", *params)

def mp_models():
    return current_connection.evaluate_single("mp-models")

def all_productions():
    return current_connection.evaluate_single("all-productions")

def buffers():
    return current_connection.evaluate_single("buffers")

def printed_visicon():
    return current_connection.evaluate_single("printed-visicon")

def print_audicon():
    return current_connection.evaluate_single("print-audicon")

def printed_audicon():
    return current_connection.evaluate_single("printed-audicon")

def printed_parameter_details(param):
    return current_connection.evaluate_single("printed-parameter-details",param)

def sorted_module_names():
    return current_connection.evaluate_single("sorted-module-names")

def modules_parameters(module):
    return current_connection.evaluate_single("modules-parameters",module)

def modules_with_parameters():
    return current_connection.evaluate_single("modules-with-parameters")

def used_production_buffers():
    return current_connection.evaluate_single("used-production-buffers")

def record_history(*params):
    return current_connection.evaluate_single("record-history",*params)

def stop_recording_history(*params):
    return current_connection.evaluate_single("stop-recording-history",*params)

def get_history_data(history,*params):
    return current_connection.evaluate_single("get-history-data",history,*params)

def history_data_available(history,file=False,*params):
    return current_connection.evaluate_single("history-data-available",history,file,*params)

def process_history_data(processor,file=False,data_params=None,processor_params=None):
    return current_connection.evaluate_single("process-history-data",file,data_params,processor_params)

def save_history_data(history,file,comment="",*params):
    return current_connection.evaluate_single("save-history-data",history,file,comment,*params)


def dm (*params):
    return current_connection.evaluate_single("dm", *params)

def sdm (*params):
    return current_connection.evaluate_single("sdm", *params)


def get_parameter_value(param):
    return current_connection.evaluate_single("get-parameter-value",param)

def set_parameter_value(param,value):
    return current_connection.evaluate_single("set-parameter-value",param,value)


def sdp (*params):
    return current_connection.evaluate_single("sdp", *params)


def simulate_retrieval_request (*spec):
    return current_connection.evaluate_single("simulate-retrieval-request", *spec)

def saved_activation_history ():
    return current_connection.evaluate_single("saved-activation-history")

def print_activation_trace (time, ms = True):
    return current_connection.evaluate_single("print-activation-trace",time,ms)

def print_chunk_activation_trace (chunk, time, ms = True):
    return current_connection.evaluate_single("print-chunk-activation-trace",chunk,time,ms)

def pp (*params):
    return current_connection.evaluate_single("pp", *params)

def trigger_reward(reward,maintenance=False):
    return current_connection.evaluate_single("trigger-reward",reward,maintenance)


def stop_output():
    current_connection.interface.no_output()

def resume_output():
    current_connection.interface.echo_output()

def hide_output():
    current_connection.interface.show_output = False

def unhide_output():
    current_connection.interface.show_output = True

def process_events():
    time.sleep(0)

def permute_list(l):

    indexes = list(range(len(l)))
    new_indexes = current_connection.evaluate_single("permute-list",indexes)
    result = []
    for i in new_indexes:
        result.append(l[i])
    return result