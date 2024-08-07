# OpenWebUI Filters and Tools

_Mirrored at Github: https://github.com/ProjectMoon/open-webui-filters_

Documentation (HTML):
[https://agnos.is/projects/open-webui-filters/][docs-html]

Documentation (Gemini):
[gemini://agnos.is/projects/open-webui-filters/][docs-gemini]

My collection of OpenWebUI Filters and Tools.

So far:

 - **Checkpoint Summarization Filter:** A work-in-progress replacement
   for the narrative memory filter for more generalized use cases.
 - **Memory Filter:** A basic narrative memory filter intended for
   long-form storytelling/roleplaying scenarios. Intended as a proof
   of concept/springboard for more advanced narrative memory.
 - **GPU Scaling Filter:** Reduce number of GPU layers in use if Ollama
   crashes due to running out of VRAM.
 - **Output Sanitization Filter:** Remove words, phrases, or
   characters from the start of model replies.
 - **OpenStreetMap Tool:** Tool for querying OpenStreetMap to look up
   address details and nearby points of interest.

## Checkpoint Sumarization Filter

A new filter for managing context use by summarizing previous parts of
the chat as the conversation continues. Designed for both general
chats and narrative/roleplay use. Work in progress.

### Configuration

There are currently 4 settings:

 - **Summarizer Model:** The model used to summarize the conversation
   as the chat continues. This must be a base model.
 - **Large Context Summarizer Model:** If large context summarization
   is turned on, use this model for summarizing huge contexts.
 - **Summarize Large Contexts:** If enabled, the filter will attempt
   to load the entire context into the large summarizer model for
   creating an initial checkpoint of an existing conversation.
 - **Wiggle Room:** This is the amount of 'wiggle room' for estimating
   a context shift. This number is subtracted from `num_ctx` for the
   purposes of determining whether or not a context shift has
   occurred.

### Usage

In general, you should only need to specify the summarizer model and
enable the filter on the OpenWebUI models that you want it to work on.
Or even enable it globally. The filter works best when used from a new
conversation, but it does have the (currently limited) ability to deal
with existing conversations.

  - When the filter detects a context shift in the conversation, it
    will summarize the pre-existing context. After that, the summary
    is appended to the system prompt, and old messages before
    summarization are dropped.
  - When the filter detects the next context shift, this process is
    repeated, and a new summarization checkpoint is created. And so
    on.

If the filter is used in an existing conversation, it will summarize
on the first time that it detects a context shift in the conversation:

 - If there are enough messages that the conversation is considered
   "big," and large context summarization is **disabled**, all but the
   last 4 messages will be dropped to form the summary.
 - If the conversation is considered "big," and large context
   summarization is **enabled**, then the large context model will be
   loaded to do the summarization, and the **entire conversation**
   will be given to it.

#### User Commands

There are some basic commands the user can use to interact with the
filter in a conversation:

 - `!nuke`: Deletes all summary checkpoints in the chat, and the
   filter will attempt to summarize from scratch the next time it
   detects a context shift.

### Limitations

There are some limitations to be aware of:
 - If you enable large context summarization, you need to make sure
   your system is capable of loading and summarizing an entire
   conversation.
 - Handling of branching conversations and regenerated responses is
   currently rather messy. It will kind of work. There are some plans
   to improve this.
 - If large context summarization is disabled, pre-existing large
   conversations will only summarize the previous 4 messages when the
   first summarization is detected.
 - The filter only loads the most recent summary, and thus the AI
   might "forget" much older information.

## Memory Filter

__Superseded By: [Checkpoint Summarization Filter][checkpoint-filter]__

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
   used, so multiple users = concurrency issues?
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

## Output Sanitization Filter

This filter is intended for models that often output unwanted
characters or terms at the beginning of replies. I have noticed this
especially with Beyonder V3 and related models. They sometimes output
a `":"` or `"Name:"` in front of replies. For example, if system prompt is
`"You are Quinn, a helpful assistant."` the model will often reply with
`"Quinn:"` as its first word.

There is one setting:

 - **Terms:** List of terms or characters to remove. This is a list,
   and in the UI, each item should be separated by a comma.

For the above example, the setting textbox should have `:,Quinn:` in
it, to remove a single colon from the start of replies, and `Quinn:`
from the start of replies.

### Other Notes

Terms are removed in the order defined by the setting. The filter
loops through each term and attempts to remove it from the start of
the LLM's reply.

## OpenStreetMap Tool

_Recommended models: Llama 3.1, Mistral Nemo Instruct._

A tool that can find certain points of interest (POIs) nearby a
requested address or place.

There are currently four settings:
 - **User Agent:** The custom user agent to set for OSM and Overpass
   Turbo API requests.
 - **From Header:** The email address for the From header for OSM and
   Overpass API requests.
 - **Nominatim API URL:** URL of the API endpoint for Nominatim, the
   reverse geocoding (address lookup) service. Defaults to the public
   instance.
 - **Overpass Turbo API URL:** URL of the API endpoint for Overpass
   Turbo, for searching OpenStreetMap. Defaults to the public
   endpoint.

The tool **will not run** without the User Agent and From headers set.
This is because the public instance of the Nominatim API will block
you if you do not set these. Use of the public Nominatim instance is
governed by their [terms of use][nom-tou].

The default API services are suitable for applications with a low
volume of traffic (absolute max 1 API call per second). If you are
running a production service, you should set up your own Nominatim and
Overpass services with caching.

# License

<img src="./agplv3.png" alt="AGPLv3" />

All filters are licensed under [AGPL v3.0+][agpl]. The code is free
software, that you can run, redistribute, modify, study, and learn
from as you see fit, as long as you extend that same freedom to
others, in accordance with the terms of the AGPL. Make sure you are
aware how this might affect your OpenWebUI deployment, if you are
deploying OpenWebUI in a public environment!

[agpl]: https://www.gnu.org/licenses/agpl-3.0.en.html
[checkpoint-filter]: #checkpoint-summarization-filter
[nom-tou]: https://operations.osmfoundation.org/policies/nominatim/
[docs-html]: https://agnos.is/projects/open-webui-filters/
[docs-gemini]: gemini://agnos.is/projects/open-webui-filters/
