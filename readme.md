# OpenWebUI Filters

My collection of OpenWebUI Filters.

So far:

 - **Memory Filter:** A basic narrative memory filter intended for
   long-form storytelling/roleplaying scenarios. Intended as a proof
   of concept/springboard for more advanced narrative memory.
 - **GPU Scaling Filter:** Reduce number of GPU layers in use if Ollama
   crashes due to running out of VRAM.

## Memory Filter

Super hacky, very basic automatic narrative memory filter for
OpenWebUI, that may or may not actually enhance narrative generation!

This is intended to be a springboard for a better, more comprehensive
filter that can coherently keep track(ish?) of plot and character
developments in long form story writing/roleplaying scenarios, where
context window length is limited (or ollama crashes on long context
length models despite having 40 GB of unused memory!).

### Configuration

The filter exposes two settings:

 - **Summarization model:** This is the model used for extracting and
   creating all of the narrative memory, and searching info. It must
   be good at following instructions. I use Gemma 2.
     - **It must be a base model.** If it's not, things will not work.
     - If you don't set this, the filter will attempt to use the model
       in the conversation. It must still be a base model.
 - **Number of messages to retain:** Number of messages to retain for the
   context. All messages before that are dropped in order to manage
   context length.

Ideally, the summarization model is the same model you are using for
the storytelling. Otherwise you may have lots of model swap-outs.

The filter hooks in to OpenWebUI's RAG settings to generate embeddings
and query the vector database. The filter will use the same embedding
model and ChromaDB instance that's configured in the admin settings.

### Usage

Enable the filter on a model that you want to use to generate stories.
It is recommended, although not required, that this be the same model
as the summarizer model (above). If you have lots of VRAM or are very
patient, you can use different models.

User input is pre-processed to 'enrich' the narrative. Replies from
the language model are analyzed post-delivery to update the story's
knowlege repository.

You will see status indicators on LLM messages indicating what the
filter is doing.

Do not reply while the model is updating its knowledge base or funny
things might happen.

### Function

What does it do?
 - When receiving user input, generate search queries for vector DB
   based on user input + last model response.
 - Search vector DB for theoretically relevant character and plot
   information.
 - Ask model to summarize results into coherent and more relevant
   stuff.
 - Inject results as <context>contextual info</context> for the model.
 - After receiving model narrative reply, generate character and plot
   info and stick them into the vector DB.

### Limitations and Known Issues

What does it not do?
 - Handle conversational branching/regeneration. In fact, this will
   pollute the knowledgebase with extra information!
   - Bouncing around some ideas to fix this. Basically requires
     building a "canonical" branching story path in the database?
 - Proper context "chapter" summarization (planned to change).
 - ~~Work properly when switching conversations due to OpenWebUI
   limitations. The chat ID is not available on incoming requests for
   some reason, so a janky workaround is used when processing LLM
   responses.~~ Fixed! (but still in a very hacky way)
 - Clear out information of old conversations or expire irrelevant
   data.

Other things to do or improve:
 - Set a minimum search score, to prevent useless stuff from coming up.
 - Figure out how to expire or update information about characters and
   events, instead of dumping it all into the vector DB.
 - Improve multi-user handling. Should technically sort of work due to
   messages having UUIDs, but is a bit messy. Only one collection is
   used, so multiple users = concurrency issues.
 - Block user input while updating the knowledgebase.

## GPU Scaling Filter

This is a simple filter that reduces the number of GPU layers in use
by Ollama when it detects that Ollama has crashed (via empty response
coming in to OpenWebUI). Right now, the logic is very basic, just
using static numbers to reduce GPU layer counts. It doesn't take into
account the number of layers in models or dynamically monitor VRAM
use.

There are three settings:
 - **Initial Reduction:** Number of layers to immediately set when an
   Ollama crash is detected. Defaults to 20.
 - **Scaling Step:** Number of layers to reduce by on subsequent crashes
   (down to a minimum of 0, i.e. 100% CPU inference). Defaults to 5.
 - **Show Status:** Whether or not to inform the user that the
   conversation is running slower due to GPU layer downscaling.

# License

AGPL v3.0+.
